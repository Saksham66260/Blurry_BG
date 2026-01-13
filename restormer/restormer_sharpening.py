import os
import cv2
import torch
import numpy as np
from restormer_arch import Restormer

# =============================
# CONFIG
# =============================
# path to background image (relative to project root)
INPUT_IMAGE = "background.png"


# output path (inside restormer folder)
OUTPUT_IMAGE = "background_restored.png"

# robust checkpoint path (works from anywhere)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "model_zoo", "defocus_deblurring.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================
# HELPER: pad image to multiple of 8
# =============================
def pad_to_multiple(img, multiple=8):
    h, w, c = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    img_padded = np.pad(
        img,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect"
    )
    return img_padded, h, w


# =============================
# LOAD IMAGE
# =============================
img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise FileNotFoundError(f"cannot load image: {INPUT_IMAGE}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0

# pad image (VERY IMPORTANT)
img_pad, orig_h, orig_w = pad_to_multiple(img, multiple=8)

# to tensor
inp = torch.from_numpy(img_pad).permute(2, 0, 1).unsqueeze(0).to(device)


# =============================
# LOAD RESTORMER MODEL
# =============================
model = Restormer(
    inp_channels=3,
    out_channels=3,
    dim=48,
    num_blocks=[4, 6, 6, 8],
    num_refinement_blocks=4,
    heads=[1, 2, 4, 8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type="WithBias"
)

checkpoint = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(checkpoint["params"])
model.to(device)
model.eval()


# =============================
# INFERENCE
# =============================
with torch.no_grad():
    restored = model(inp)

# to numpy
restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()

# crop back to original size
restored = restored[:orig_h, :orig_w, :]

# back to uint8
restored = np.clip(restored * 255.0, 0, 255).astype(np.uint8)

# save
cv2.imwrite(OUTPUT_IMAGE, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))

print("✅ Restormer background restoration complete")
print(f"Saved to: {OUTPUT_IMAGE}")
