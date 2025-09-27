# 🍔 FoodVision – Transfer Learning with EfficientNet

This project applies **transfer learning** with **EfficientNet** (from `torchvision.models`) to classify images in the **FoodVision dataset**.  
The goal is to leverage pretrained ImageNet weights and fine-tune the final layers for food classification.

---

## 📌 Project Overview
- Uses **EfficientNet-B0** pretrained on ImageNet.
- Replaces the final classifier layer with a 3-class output (customizable).
- Trains on the **FoodVision dataset** (organized with `torchvision.datasets.ImageFolder`).
- Tracks experiments with **TensorBoard** (scalars, histograms, graphs, images).
- Implements a modular **train/test loop** with accuracy and loss reporting.

---

## ⚙️ Features
- **Transfer Learning**: Load pretrained weights, freeze backbone, fine-tune classifier.
- **Experiment Tracking**: TensorBoard integration for scalars, histograms, and graphs.
- **Flexible Training Loop**: `train_step()` and `test_step()` functions for cleaner code.
- **Image Logging**: Optionally log sample images and predictions to TensorBoard.
- **Reproducibility**: Timestamped run folders and experiment logs.

---
