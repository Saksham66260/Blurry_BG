import os
import cv2
import time
import numpy as np
import torch
import subprocess
from threading import Thread, Lock
from queue import Queue

# =====================
# SAFE DEVICE SELECTOR (CUDA -> MPS -> CPU)
# =====================
def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_best_device()
print(" Using device:", DEVICE)

# =====================
# CONFIG
# =====================
CAPTURE_PIPELINE = "pipeline.py"

YOLO_MODEL = "yolo11n-seg.pt"
YOLO_CONF = 0.5
YOLO_SIZE = 320
YOLO_EVERY_N = 12

RESTORMER_CKPT = os.path.join("restormer", "model_zoo", "defocus_deblurring.pth")
RESTORMER_SIZE = 384
RESTORMER_EVERY_N = 15

MASK_BLUR = 31
MASK_SMOOTH = 0.8

# =====================
# SAFE IMPORTS
# =====================
HAS_YOLO = False
HAS_RESTORMER = False
yolo = None
restormer = None

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception as e:
    print("⚠ YOLO not available:", e)

try:
    from restormer.restormer_arch import Restormer
    HAS_RESTORMER = True
except Exception as e:
    print("⚠ Restormer not available:", e)

# =====================
# LOAD YOLO
# =====================
if HAS_YOLO and os.path.exists(YOLO_MODEL):
    try:
        print("[1] Loading YOLO...")
        yolo = YOLO(YOLO_MODEL)
        print("✓ YOLO loaded")
    except Exception as e:
        print("⚠ YOLO load failed:", e)
        HAS_YOLO = False
else:
    print("⚠ YOLO model not found, masking disabled.")
    HAS_YOLO = False

# =====================
# LOAD RESTORMER
# =====================
if HAS_RESTORMER and os.path.exists(RESTORMER_CKPT):
    try:
        print("[2] Loading Restormer...")
        restormer = Restormer()
        ckpt = torch.load(RESTORMER_CKPT, map_location="cpu")
        restormer.load_state_dict(ckpt["params"] if "params" in ckpt else ckpt, strict=False)
        restormer = restormer.to(DEVICE).eval()
        print("✓ Restormer loaded")
    except Exception as e:
        print("⚠ Restormer load failed:", e)
        HAS_RESTORMER = False
else:
    print("⚠ Restormer checkpoint not found, using fallback enhancement.")
    HAS_RESTORMER = False

# =====================
# CAMERA OPEN (GENERIC)
# =====================
def open_camera():
    # Covers common ranges across Windows/Linux/Mac
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✓ Camera {i} opened: {frame.shape[1]}x{frame.shape[0]}")
                return cap
            cap.release()
    return None

# =====================
# MASK USING YOLO (GENERIC)
# =====================
def get_mask_yolo(img_rgb):
    """
    Returns foreground mask (1=FG, 0=BG).
    If YOLO not available, return zeros => treat everything as background.
    """
    h, w = img_rgb.shape[:2]

    if not HAS_YOLO or yolo is None:
        return np.zeros((h, w), dtype=np.float32)

    small = cv2.resize(img_rgb, (YOLO_SIZE, YOLO_SIZE))
    results = yolo.predict(source=small, conf=YOLO_CONF, verbose=False)[0]

    if results.masks is None or len(results.masks.data) == 0:
        return np.zeros((h, w), dtype=np.float32)

    masks = results.masks.data.detach().cpu().numpy()

    # use largest instance
    if len(masks) > 1:
        areas = [m.sum() for m in masks]
        mask = masks[int(np.argmax(areas))]
    else:
        mask = masks[0]

    mask = cv2.resize(mask.astype(np.float32), (w, h))
    mask = (mask > 0.5).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (MASK_BLUR, MASK_BLUR), 0)
    return np.clip(mask, 0.0, 1.0)

# =====================
# BACKGROUND ENHANCEMENT
# =====================
def fallback_enhance(img_rgb):
    """
    Generic enhancement that works everywhere:
    - mild unsharp + contrast
    """
    blur = cv2.GaussianBlur(img_rgb, (0, 0), 1.2)
    sharp = cv2.addWeighted(img_rgb, 1.6, blur, -0.6, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def deblur_restormer(img_rgb):
    if not HAS_RESTORMER or restormer is None:
        return fallback_enhance(img_rgb)

    h, w = img_rgb.shape[:2]

    scale = RESTORMER_SIZE / max(h, w)
    if scale < 1:
        small = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = img_rgb.copy()

    tensor = torch.from_numpy(small).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        out = restormer(tensor)

    out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)

    if scale < 1:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

    return out

# =====================
# BLENDING (foreground kept original)
# =====================
def blend_fg_bg(original, mask, enhanced):
    mask3 = np.repeat(mask[:, :, None], 3, axis=2).astype(np.float32)

    # soften edge
    mask3 = cv2.GaussianBlur(mask3, (21, 21), 0)

    # fg = original, bg = enhanced
    result = original.astype(np.float32) * mask3 + enhanced.astype(np.float32) * (1.0 - mask3)
    return np.clip(result, 0, 255).astype(np.uint8)

# =====================
# BACKGROUND THREAD
# =====================
class BackgroundProcessor:
    def __init__(self):
        self.input_queue = Queue(maxsize=1)
        self.mask_queue = Queue(maxsize=1)
        self.enh_queue = Queue(maxsize=1)

        self.latest_mask = None
        self.latest_enh = None

        self.lock = Lock()
        self.running = True
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        frame_count = 0
        while self.running:
            try:
                if self.input_queue.empty():
                    time.sleep(0.01)
                    continue

                frame_rgb = self.input_queue.get(timeout=0.1)

                # Mask every N
                if frame_count % YOLO_EVERY_N == 0:
                    try:
                        mask = get_mask_yolo(frame_rgb)
                        with self.lock:
                            self.latest_mask = mask
                        if self.mask_queue.full():
                            try: self.mask_queue.get_nowait()
                            except: pass
                        self.mask_queue.put(mask)
                    except Exception as e:
                        print("Mask error:", e)

                # Enhancement every N
                if frame_count % RESTORMER_EVERY_N == 0:
                    try:
                        enh = deblur_restormer(frame_rgb)
                        with self.lock:
                            self.latest_enh = enh
                        if self.enh_queue.full():
                            try: self.enh_queue.get_nowait()
                            except: pass
                        self.enh_queue.put(enh)
                    except Exception as e:
                        print("Enhance error:", e)

                frame_count += 1

            except Exception as e:
                print("Processing error:", e)
                time.sleep(0.05)

    def submit(self, frame_rgb):
        if self.input_queue.full():
            try: self.input_queue.get_nowait()
            except: pass
        try:
            self.input_queue.put_nowait(frame_rgb.copy())
        except:
            pass

    def get_mask(self):
        try:
            return self.mask_queue.get_nowait()
        except:
            with self.lock:
                return self.latest_mask

    def get_enhanced(self):
        try:
            return self.enh_queue.get_nowait()
        except:
            with self.lock:
                return self.latest_enh

    def stop(self):
        self.running = False
        self.thread.join()

# =====================
# MAIN LOOP
# =====================
print("[3] Starting preview...")
cap = open_camera()
if cap is None:
    raise RuntimeError(" Camera not available")

processor = BackgroundProcessor()

frame_id = 0
processing_enabled = True
IS_CAPTURING = False
view_mode = 2  # default: selective

prev_time = time.time()
fps_hist = []

current_mask = None
prev_mask = None
current_enh = None

print("=" * 70)
print("CONTROLS:")
print("  '1' - Original vs Enhanced(full)")
print("  '2' - Original vs Selective Background Enhanced")
print("  '3' - Show Foreground Mask")
print("  SPACE - Toggle processing ON/OFF")
print("  's' - Save comparison")
print("  'c' - CAPTURE (run full pipeline)")
print("  'q' - QUIT")
print("=" * 70)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        if processing_enabled and not IS_CAPTURING:
            processor.submit(frame_rgb)

        new_mask = processor.get_mask()
        if new_mask is not None:
            if prev_mask is None:
                current_mask = new_mask
            else:
                current_mask = MASK_SMOOTH * prev_mask + (1 - MASK_SMOOTH) * new_mask
            prev_mask = current_mask

        new_enh = processor.get_enhanced()
        if new_enh is not None:
            current_enh = new_enh

        if current_mask is None:
            current_mask = np.zeros((h, w), dtype=np.float32)
        if current_enh is None:
            current_enh = frame_rgb.copy()

        selective = blend_fg_bg(frame_rgb, current_mask, current_enh)

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        fps_hist.append(fps)
        if len(fps_hist) > 20:
            fps_hist.pop(0)
        avg_fps = sum(fps_hist) / len(fps_hist)

        left = frame_rgb.copy()
        if view_mode == 1:
            right = current_enh
            txt = "FULL FRAME ENHANCED"
        elif view_mode == 2:
            right = selective
            txt = "SELECTIVE BG ENHANCED"
        else:
            mask_vis = (current_mask * 255).astype(np.uint8)
            right = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2RGB)
            txt = "FG MASK (WHITE=FG)"

        cv2.putText(left, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(right, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(right, f"FPS: {avg_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        status = "ON" if processing_enabled else "OFF"
        cv2.putText(right, f"Processing: {status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        combined = np.hstack([left, right])
        cv2.imshow("Samsung Blur BG Enhancement Preview", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            processing_enabled = not processing_enabled
            print("Processing:", "ON" if processing_enabled else "OFF")
        elif key == ord("1"):
            view_mode = 1
        elif key == ord("2"):
            view_mode = 2
        elif key == ord("3"):
            view_mode = 3
        elif key == ord("s"):
            name = f"comparison_{int(time.time())}.png"
            cv2.imwrite(name, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            print("✓ Saved:", name)
        elif key == ord("c") and not IS_CAPTURING:
            IS_CAPTURING = True
            print("\n CAPTURING...")
            ret_cap, frame_cap = cap.read()
            if ret_cap:
                cv2.imwrite("captured_input.png", frame_cap)
                print("✓ Saved captured_input.png")
                subprocess.run(["python3", CAPTURE_PIPELINE])
            IS_CAPTURING = False

        frame_id += 1

finally:
    processor.stop()
    cap.release()
    cv2.destroyAllWindows()
    print(" DONE")
