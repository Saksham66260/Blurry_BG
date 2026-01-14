import os
import gdown
from ultralytics import YOLO

# -------------------------
# Restormer weights (Drive)
# -------------------------
FILE_ID = "1Ls8ep2FKjscIXde33r3lXK-gQ5TcgTEu"
url = f"https://drive.google.com/uc?id={FILE_ID}"

restormer_path = "restormer/model_zoo/defocus_deblurring.pth"
os.makedirs(os.path.dirname(restormer_path), exist_ok=True)

print("Downloading Restormer weights...")
gdown.download(url, restormer_path, quiet=False)
print("Restormer ready.")

# -------------------------
# YOLOv11 Segmentation model
# -------------------------
print("Downloading YOLOv11 segmentation model...")
YOLO("yolo11n-seg.pt")  # auto-downloads
print("YOLOv11 segmentation ready.")

print("\n All model weights ready!")
