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
