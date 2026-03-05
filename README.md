# Computer Vision with Deep Learning (PyTorch)

## Project Overview

This project contains two computer vision tasks implemented in PyTorch:

1. **Modern CNN ablation study** using ConvNeXt-Tiny for image classification
2. **Object detection comparison** using FCOS and RetinaNet

The goal is to study both modern CNN design choices and object detection frameworks in real training pipelines.

---

## Part 1: ConvNeXt Ablation Study

In the first part, I fine-tuned **ConvNeXt-Tiny** on an image classification task and evaluated how different architectural components affect validation performance.

### Model
- ConvNeXt-Tiny
- Pretrained ImageNet weights
- Fine-tuned in PyTorch / Torchvision

### Ablation Settings
The following ablations were tested:

- LayerNorm → BatchNorm
- GELU → ReLU
- 7×7 depthwise convolution → 3×3 standard convolution
- Inverted bottleneck → standard bottleneck
- Remove stochastic depth

### Result Summary
The baseline ConvNeXt-Tiny achieved approximately **85.5% validation accuracy**.  
Among all ablations, replacing the **7×7 depthwise convolution** caused the largest performance drop, showing that large-kernel convolution was the most important design component in this experiment.

---

## Part 2: Object Detection with FCOS and RetinaNet

In the second part, I implemented an object detection pipeline comparing:

- **FCOS** (anchor-free detector)
- **RetinaNet** (anchor-based detector)

### Dataset
- NWPU VHR-10 aerial object detection dataset

### Detection Pipeline
The detection workflow includes:

- dataset loading
- image resizing
- random horizontal flip
- color jitter
- bounding-box preprocessing with `torchvision.transforms.v2`
- model training and validation
- mAP@0.5 evaluation

### Models Used
- `fcos_resnet50_fpn`
- `retinanet_resnet50_fpn`

This project compares anchor-free and anchor-based detection strategies on aerial imagery.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Repository Structure

```text
computer-vision-with-dl/
├── Compter_Vision_with_DL.ipynb
├── README.md
