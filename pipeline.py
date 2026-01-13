import torch
import cv2
import numpy as np
import time
import os
from PIL import Image

# ======================
# CONFIG
# ======================
INPUT_IMAGE = "input.jpg"  # Path to input image
RESTORMER_CKPT = "restormer/model_zoo/defocus_deblurring.pth"

# ======================
# DEVICE SETUP (IMPORTANT)
# ======================
# Depth (MiDaS) can run on MPS safely
DEPTH_DEVICE = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"

# Restormer MUST run on CPU on Mac (to avoid MPS OOM crash)
RESTORE_DEVICE = "cpu"

print(f"✓ Depth on: {DEPTH_DEVICE}")
print(f"⚠ Restormer forced on: {RESTORE_DEVICE} (prevents MPS OOM)")

# ======================
# PARAMETERS
# ======================
SPEED_MODE = "fast"
MAX_DIMENSION = 768  # safe + good quality (1024 causes higher compute)

# Foreground detection thresholds
FG_DETECTION_THRESHOLD = 0.15  # Depth variance threshold
FG_MIN_AREA_RATIO = 0.05       # Minimum 5% of image should be foreground

# Depth thresholds for segmentation (used only if FG detected)
DEPTH_THRESHOLD_LOW = 0.35
DEPTH_THRESHOLD_HIGH = 0.65

# Blending parameters
MASK_BLUR_SIZE = 11
EDGE_BLUR_SIZE = 5
ENHANCE_SHARPNESS = True

print(f"Speed mode: {SPEED_MODE}")
print(f"Max dimension: {MAX_DIMENSION}px")

# ======================
# LOAD RESTORMER (CPU ONLY)
# ======================
print("\n[1/4] Loading Restormer...")
from restormer.restormer_arch import Restormer

model = Restormer()

# Always load checkpoint on CPU (stable)
checkpoint = torch.load(RESTORMER_CKPT, map_location="cpu")
if "params" in checkpoint:
    model.load_state_dict(checkpoint["params"])
elif "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(RESTORE_DEVICE).eval()
print("✓ Restormer loaded (CPU)")

# ======================
# READ IMAGE
# ======================
print("\n[2/4] Loading image...")
img_bgr = cv2.imread(INPUT_IMAGE)
if img_bgr is None:
    raise FileNotFoundError(f"Cannot read: {INPUT_IMAGE}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
orig_h, orig_w = img_rgb.shape[:2]
print(f"  Original size: {orig_w}x{orig_h}")

img_rgb_original = img_rgb.copy()

# Resize for processing
scale = min(1.0, MAX_DIMENSION / max(orig_h, orig_w))
if scale < 1.0:
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(f"  Processing at: {new_w}x{new_h} (scale: {scale:.2f})")

h, w = img_rgb.shape[:2]

# ======================
# FOREGROUND DETECTION (MiDaS)
# ======================
print("\n[3/4] Detecting foreground...")

print("  Loading MiDaS...")
if SPEED_MODE == "fast":
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
else:
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
    midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).dpt_transform

midas.to(DEPTH_DEVICE).eval()

# Estimate depth
input_batch = midas_transform(img_rgb).to(DEPTH_DEVICE)

with torch.no_grad():
    depth = midas(input_batch)
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

# Bring depth to CPU numpy
depth = depth.detach().cpu().numpy()
depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

# Analyze depth map to detect foreground
depth_std = float(np.std(depth))
depth_range = float(np.max(depth) - np.min(depth))

print(f"  Depth std: {depth_std:.3f}, range: {depth_range:.3f}")

# Method 1: Check depth variance
has_depth_variation = depth_std > FG_DETECTION_THRESHOLD

# Method 2: Bimodal distribution (two peaks = fg + bg)
hist, bins = np.histogram(depth.flatten(), bins=50)
hist_smooth = cv2.GaussianBlur(hist.reshape(-1, 1), (5, 1), 0).flatten()
peaks = []
for i in range(1, len(hist_smooth) - 1):
    if hist_smooth[i] > hist_smooth[i - 1] and hist_smooth[i] > hist_smooth[i + 1]:
        if hist_smooth[i] > 0.1 * np.max(hist_smooth):
            peaks.append(i)

has_bimodal = len(peaks) >= 2

# Method 3: Foreground area ratio check
depth_binary = (depth < 0.4).astype(np.uint8)
fg_area_ratio = float(np.sum(depth_binary) / (h * w))
has_fg_area = fg_area_ratio > FG_MIN_AREA_RATIO and fg_area_ratio < 0.95

# Combine detection methods
HAS_FOREGROUND = has_depth_variation and (has_bimodal or has_fg_area)

print(f"  Depth variation: {'Yes' if has_depth_variation else 'No'}")
print(f"  Bimodal distribution: {'Yes' if has_bimodal else 'No'} ({len(peaks)} peaks)")
print(f"  Foreground area: {fg_area_ratio * 100:.1f}% ({'Yes' if has_fg_area else 'No'})")
print(f"\n  → Foreground detected: {'YES' if HAS_FOREGROUND else 'NO'}")

# free MiDaS
del midas

# ======================
# PROCESS IMAGE
# ======================
print(f"\n[4/4] Processing image...")

def pad_image(img, multiple=8):
    hh, ww = img.shape[:2]
    pad_h = (multiple - hh % multiple) % multiple
    pad_w = (multiple - ww % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return img, hh, ww

if HAS_FOREGROUND:
    print("  Mode: SELECTIVE DEBLUR (foreground detected)")
    start = time.time()

    # Create mask
    mask = np.clip((depth - DEPTH_THRESHOLD_LOW) / (DEPTH_THRESHOLD_HIGH - DEPTH_THRESHOLD_LOW), 0, 1)
    mask = cv2.GaussianBlur(mask, (MASK_BLUR_SIZE, MASK_BLUR_SIZE), 0)
    mask_3c = np.repeat(mask[:, :, None], 3, axis=2)

    # Extract background
    background = img_rgb * (1 - mask_3c)

    # Pad and process background
    bg_padded, orig_h_pad, orig_w_pad = pad_image(background)
    bg_tensor = torch.from_numpy(bg_padded).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    bg_tensor = bg_tensor.to(RESTORE_DEVICE)

    print("    Running Restormer on background (CPU)...")
    with torch.no_grad():
        restored = model(bg_tensor)

    # Post-process
    restored = restored[:, :, :orig_h_pad, :orig_w_pad]
    restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255
    restored = restored.astype(np.uint8)

    # Blend with edge-aware alpha
    if ENHANCE_SHARPNESS:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edges_norm = edges.astype(np.float32) / 255.0

        alpha = mask.copy()
        alpha_smooth = cv2.GaussianBlur(alpha, (EDGE_BLUR_SIZE, EDGE_BLUR_SIZE), 0)
        alpha = alpha_smooth * (1 - edges_norm) + alpha * edges_norm
        alpha = alpha[:, :, None]

        final = (img_rgb * alpha + restored * (1 - alpha)).astype(np.uint8)

        # Optional unsharp mask on background
        bg_mask_binary = (mask < 0.25).astype(np.uint8)
        if bg_mask_binary.sum() > 1000:
            blurred = cv2.GaussianBlur(final, (3, 3), 1.0)
            sharpened = cv2.addWeighted(final, 1.6, blurred, -0.6, 0)
            bg_mask_3c = np.repeat(bg_mask_binary[:, :, None], 3, axis=2)
            final = final * (1 - bg_mask_3c) + sharpened * bg_mask_3c
            final = final.astype(np.uint8)
    else:
        alpha = cv2.GaussianBlur(mask, (EDGE_BLUR_SIZE, EDGE_BLUR_SIZE), 0)[:, :, None]
        final = (img_rgb * alpha + restored * (1 - alpha)).astype(np.uint8)

    print(f"  ✓ Selective deblur complete ({time.time() - start:.2f}s)")

else:
    print("  Mode: FULL DEBLUR (no foreground detected)")
    start = time.time()

    # Pad and process entire image
    img_padded, orig_h_pad, orig_w_pad = pad_image(img_rgb)
    img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(RESTORE_DEVICE)

    print("    Running Restormer on full image (CPU)...")
    with torch.no_grad():
        restored = model(img_tensor)

    # Post-process
    restored = restored[:, :, :orig_h_pad, :orig_w_pad]
    final = restored.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255
    final = final.astype(np.uint8)

    restored = final.copy()
    mask = np.ones((h, w), dtype=np.float32)

    print(f"  ✓ Full deblur complete ({time.time() - start:.2f}s)")

# ======================
# UPSCALE TO ORIGINAL
# ======================
if scale < 1.0:
    print(f"  Upscaling to original size: {orig_w}x{orig_h}")
    final = cv2.resize(final, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
    depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    restored = cv2.resize(restored, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

# ======================
# SAVE OUTPUTS
# ======================
print("\n[Saving outputs...]")

cv2.imwrite("final_output.png", cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
cv2.imwrite("depth_map.png", (depth * 255).astype(np.uint8))
cv2.imwrite("blur_mask.png", (mask * 255).astype(np.uint8))
cv2.imwrite("restored_bg.png", cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))

comparison = np.hstack([img_rgb_original, final])
cv2.imwrite("comparison.png", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

comparison2 = np.hstack([img_rgb_original, restored, final])
cv2.imwrite("comparison_all.png", cv2.cvtColor(comparison2, cv2.COLOR_RGB2BGR))

print("✓ Outputs saved:")
print("  - final_output.png")
print("  - comparison.png")
print("  - comparison_all.png (input | restored | final)")
print("  - depth_map.png")
print("  - blur_mask.png")
print("  - restored_bg.png")

# Save detection info
with open("detection_info.txt", "w") as f:
    f.write("Foreground Detection Results\n")
    f.write("============================\n\n")
    f.write(f"Processing mode: {'SELECTIVE DEBLUR' if HAS_FOREGROUND else 'FULL DEBLUR'}\n")
    f.write(f"Foreground detected: {HAS_FOREGROUND}\n\n")
    f.write("Detection metrics:\n")
    f.write(f"  - Depth std: {depth_std:.3f} (threshold: {FG_DETECTION_THRESHOLD})\n")
    f.write(f"  - Depth range: {depth_range:.3f}\n")
    f.write(f"  - Histogram peaks: {len(peaks)}\n")
    f.write(f"  - Foreground area: {fg_area_ratio * 100:.1f}%\n")

print("  - detection_info.txt")

print("\n" + "=" * 50)
print(f"PROCESSING MODE: {'SELECTIVE' if HAS_FOREGROUND else 'FULL'}")
print("=" * 50)