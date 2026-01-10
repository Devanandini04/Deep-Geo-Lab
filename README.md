# 🌍 Deep-Geo-Lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Learning-success)]()

> **Bridging the gap between Deep Learning mathematical foundations and Geospatial Intelligence.**

## 🚀 About The Project
This repository documents my structured journey into **GeoAI**. Instead of relying solely on high-level APIs, I am building algorithms from scratch to understand the mathematics behind the models, before applying them to satellite imagery.

**Core Philosophy:** *Code the Math → Build the Model → Analyze the Earth.*

---

## 🗺️ Learning Roadmap

### Phase 1: The Mathematical Core (Current Focus) 🧠
- [x] **Perceptrons:** Implementing the logic of a single neuron.
- [x] **Multi-Layer Perceptron (MLP):** Coding Forward & Backward Propagation.
- [x] **Optimization:** Implementing Gradient Descent & Loss Functions manually.

### Phase 2: Computer Vision 👁️
- [x] **CNN Architectures:** Building ConvNets for feature extraction.
- [ ] **Object Detection:** Understanding bounding boxes and anchors.

### Phase 3: GeoAI Integration 🛰️
- [ ] **Rasterio Basics:** Handling `.tif` satellite data & Coordinate Reference Systems (CRS).
- [ ] **Semantic Segmentation:** Applying U-Net to detect land/water bodies from space.

---

## 🛠️ Tech Stack

| Domain | Tools & Libraries |
| :--- | :--- |
| **Languages** | ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) |
| **Deep Learning** | `NumPy` (Math), `TensorFlow/Keras` (Upcoming), `PyTorch` |
| **Geospatial** | `Rasterio` (Pixel Data), `GDAL` (Transforms), `GeoPandas` |
| **Visualization** | `Matplotlib`, `Seaborn` |

---

## 📂 Repository Structure
```text
Deep-Geo-Lab/
├── 01-ANN-Foundations/      # Neural Networks from Scratch (No Keras)
├── 02-CNN-Vision/           # Image Processing Logic
├── Parallel-Track-Rasterio/ # Satellite Data Processing Experiments
└── README.md
