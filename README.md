# Blurry Background Enhancement using Deep Learning

A deep learning–based pipeline to enhance sharp foreground objects while intelligently handling blurry or defocused backgrounds.  
This project focuses on **image quality improvement**, **sharpness analysis**, and **defocus-aware enhancement** using modern computer vision techniques.

---

## 🔍 Project Overview

Blurry or defocused backgrounds are common in images captured under:
- low-light conditions  
- shallow depth-of-field  
- motion blur  
- camera focus limitations  

This project explores a **learning-based approach** to improve perceptual quality by:
- analyzing image sharpness
- enhancing defocused regions
- preserving foreground details

The pipeline is designed to be **modular**, **experiment-friendly**, and suitable for further research or deployment.

---

## 🧠 Key Features

- 📌 Sharpness and blur-aware processing  
- 📌 Deep learning–based enhancement (Restormer-based architecture)  
- 📌 Modular pipeline for easy experimentation  
- 📌 Clean evaluation-ready structure  
- 📌 No unnecessary binaries or environments committed  

---

## 🏗️ Repository Structure

Blurry_BG/
│
├── restormer/
│ ├── restormer_arch.py # Model architecture
│ ├── restormer_sharpening.py # Enhancement logic
│ └── model_zoo/ # (weights excluded from git)
│
├── pipeline.py # End-to-end processing pipeline
├── metrics.py # Image quality metrics
├── requirements.txt # Dependencies
├── .gitignore # Clean repo rules
└── README.md

---

## 📦 Model Weights

This project does **not** include pretrained model weights in the repository to keep it lightweight and GitHub-friendly.

Before running the pipeline, download the required weights using:

```bash
python download_weights.py
This will automatically place the model file at:
restormer/model_zoo/defocus_deblurring.pth
Make sure this step is completed before executing the main pipeline.

---

## 2️⃣ Update the *Usage* section (small tweak)

Change your usage section to:

```md
## ▶️ Usage

1. Download model weights:
```bash
python download_weights.py
Run the enhancement pipeline:
python pipeline.py

This makes the workflow crystal clear.

---

## 3️⃣ Update the *What’s NOT Included* section

Replace that section with:

```md
## 🚫 What’s NOT Included (by design)

- ❌ Virtual environments (`venv/`)
- ❌ Pretrained model weights (`.pth`)
- ❌ System-generated files

Model weights are downloaded separately using `download_weights.py`.
