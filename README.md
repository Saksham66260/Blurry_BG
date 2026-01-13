# Blurry Background Enhancement using Deep Learning

A deep learning–based pipeline to enhance sharp foreground objects while intelligently handling blurry or defocused backgrounds.  
This project focuses on **image quality improvement**, **sharpness analysis**, and **defocus-aware enhancement** using modern computer vision techniques.

---

##  Project Overview

Blurry or defocused backgrounds are common in images captured under:
- low-light conditions  
- shallow depth-of-field  

- camera focus limitations  

This project explores a **learning-based approach** to improve perceptual quality by:
- analyzing image sharpness
- enhancing defocused regions
- preserving foreground details

The pipeline is designed to be **modular**, **experiment-friendly**, and suitable for further research or deployment.

---

##  Key Features

-  Sharpness and blur-aware processing  
-  Deep learning–based enhancement (Restormer-based architecture)  
-  Modular pipeline for easy experimentation  
-  Clean evaluation-ready structure  
 

---

## 📁 Repository Structure


Blurry_BG/
│
├── restormer/
│   ├── restormer_arch.py        # model architecture
│   ├── restormer_sharpening.py  # enhancement logic
│   └── model_zoo/               # (weights excluded from git)
│
├── pipeline.py                  # end-to-end processing pipeline
├── metrics.py                   # image quality metrics
├── add_weights.py               # downloads pretrained model weights
├── requirements.txt             # dependencies
├── .gitignore                   # clean repo rules
└── README.md

---

##  Model Weights

This project does **not** include pretrained model weights in the repository to keep it lightweight and GitHub-friendly.

Before running the pipeline, download the required weights using:


python add_weights.py
This will automatically place the model file at:
restormer/model_zoo/defocus_deblurring.pth
Make sure this step is completed before executing the main pipeline.

---


##  Usage

1. Download model weights: python add_weights.py

2. Run the enhancement pipeline: python pipeline.py

3. Run metrics.py to get detailed analysis(might take some time)



---


Model weights are downloaded separately using `add_weights.py`.
