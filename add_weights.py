import os
import urllib.request

url = "https://drive.google.com/drive/folders/1zkRQZRQQNf0ag2rvMMvWYURl_s66DcfU?usp=sharing"
save_path = "restormer/model_zoo/defocus_deblurring.pth"

os.makedirs(os.path.dirname(save_path), exist_ok=True)
urllib.request.urlretrieve(url, save_path)

print("model weights downloaded successfully")
