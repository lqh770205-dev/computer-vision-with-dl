# 🧠 Computer Vision with Deep Learning

I built a set of **computer vision systems and experiments** in PyTorch, focusing on:

- CNN architecture design (ConvNeXt ablation study)
- Modern object detection pipelines (FCOS vs RetinaNet)

This project emphasizes **understanding model design choices and their impact on performance**, rather than just applying pre-built models.

---

## 🚀 Key Highlights

- Designed and ran **ConvNeXt architecture ablation experiments**
- Built a full **object detection training + evaluation pipeline**
- Implemented **anchor-based vs anchor-free detector comparison**
- Developed custom components:
  - dataset loaders (COCO-style + fallback)
  - optimizer with differential learning rates
  - NMS + post-processing
  - mAP@0.5 evaluation from scratch

---

## 🧩 Part 1 — ConvNeXt Ablation Study

I performed controlled experiments on **ConvNeXt-Tiny** to understand how architectural components affect performance.

### Ablations implemented:

- LayerNorm → BatchNorm
- GELU → ReLU
- 7×7 depthwise → 3×3 standard convolution
- Inverted bottleneck → standard bottleneck
- Removing stochastic depth

### Insight

These experiments show how modern CNN performance depends heavily on:

- normalization choice
- activation function
- receptive field design
- residual architecture

---

## 🎯 Part 2 — Object Detection System

I built a full detection pipeline comparing:

### 🔹 FCOS (Anchor-Free)
- Direct prediction from feature maps
- Simpler formulation, fewer hyperparameters

### 🔹 RetinaNet (Anchor-Based)
- FPN + anchor design
- Focal loss for class imbalance

---

## ⚙️ System Design

### Training Pipeline

- Custom dataset loader (COCO format)
- Data augmentation + resizing
- Differential learning rate optimizer:
  - backbone: 1e-4
  - detection head: 1e-2

### Inference Pipeline

- Score filtering
- Class-wise NMS
- Prediction post-processing

### Evaluation

- Implemented **mAP@0.5 from scratch**
- Per-class AP analysis

---

## 📊 Results

- Compared convergence behavior (loss curves)
- Evaluated detection quality via mAP
- Analyzed performance differences between FCOS and RetinaNet

---

## 📁 Project Structure
computer-vision-with-dl/
├── computer_vision_with_dl.ipynb
├── README.md
、
---

## 🛠️ Tech Stack

- Python
- PyTorch / Torchvision
- ConvNeXt
- FCOS / RetinaNet
- COCO-style detection pipeline

---

## 🧠 Key Takeaways

- CNN performance is highly sensitive to architectural design choices
- Anchor-free detectors simplify design but may behave differently across datasets
- Building detection systems from scratch provides deeper understanding than using APIs

---

## 👤 Author

Qiheng Li  
NYU MS Electrical & Computer Engineering  
