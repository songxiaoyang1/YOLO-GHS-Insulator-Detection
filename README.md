# YOLO-GHS: A Method for High-Precision Detection of Insulator Defects on Transmission Lines Under Severe Weather Conditions


> **Official PyTorch implementation of the core modules for the paper:** > *"YOLO-GHS: A Lightweight Object Detection Model for Insulator Defects in Adverse Weather via Spatial-Channel Exponential Attention"* (Under Review)

---

## 📌 A Note on Code Availability (Plug-and-Play)
To ensure clarity and highlight our exact methodological contributions, this repository provides the **core innovative modules** rather than a redundant clone of the entire massive standard training framework. 

This "plug-and-play" approach allows researchers to directly inspect our exact mathematical and structural modifications without navigating thousands of lines of unmodified boilerplate code. You can easily integrate our custom head and backbone configurations into the standard Ultralytics YOLO ecosystem.

---

## 📂 Repository Structure

Based on our plug-and-play philosophy, we provide the following core files:

* 📄 **`yolo-ghs.yaml`**: The complete architectural configuration of our model, detailing the `GhostHGNetV2` backbone and the custom head.
* 📄 **`SEAM.py`**: The exact PyTorch implementation of the `Detect_SEAM` module and the `Channel Exp` mathematical mechanisms.
* 📦 **`best.pt`**: The final pre-trained weights (only **5.4 MB**!). You can use this directly for evaluation or inference.
* 🖼️ **`image.png`**: Network architecture / visual reference.

---

## 💡 Key Highlights

Deploying deep learning models on Unmanned Aerial Vehicles (UAVs) for power line inspection faces two critical bottlenecks: stringent hardware constraints and severe physical occlusion caused by adverse weather. YOLO-GHS solves these exact challenges:

1. **Ultra-Lightweight for UAV Edge AI:** Achieves state-of-the-art accuracy with only **5.4G FLOPs**, **2.5M parameters**, and a tiny **5.4 MB** model size. It is inherently lighter than the YOLOv11n baseline.
2. **Feature-Level Exposure Compensation:** Introduces the novel **Detect_SEAM** head. Unlike standard attention that uses linear Sigmoid attenuation (which irreversibly erases weak features under snow/fog), our `Channel Exp` function mathematically guarantees feature preservation using exponential compensation ($W_{exp} = e^Z$). When confidence drops due to occlusion ($Z \to 0$), the weight intrinsically converges to one ($W_{exp} \to 1$), unconditionally preserving the residual features.
3. **Robust in Extreme Weather:** Achieves a massive boost in small object detection under heavy snow and dense fog.

---

## 📊 Comprehensive Performance Comparison

The following table presents the comprehensive comparative experiments on the IDID Dataset under severe weather conditions (Heavy Snow & Dense Fog). 

*Our `YOLO-GHS` achieves the highest overall accuracy and small-object detection capability while maintaining an exceptionally low memory footprint.*

| Model | P | R | mAP@0.5 | F1 | mAP@[0.5:0.95] | $AP_S$ | FLOPs (G) | FPS | Params (M) | Size (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SSD | 0.805 | 0.852 | 0.868 | 0.828 | 0.584 | 0.395 | 38.6 | 43.3 | 24.3 | 92.6 |
| Faster-RCNN | 0.885 | 0.890 | 0.912 | 0.887 | 0.672 | 0.512 | 28.7 | 57.5 | 28.5 | 108.2 |
| YOLOv5s | 0.862 | 0.875 | 0.884 | 0.868 | 0.615 | 0.463 | 16.8 | 48.7 | 7.0 | 14.1 |
| YOLOv8s | 0.895 | 0.892 | 0.915 | 0.893 | 0.688 | 0.536 | 40.5 | 78.7 | 11.2 | 22.5 |
| YOLOv10 | 0.912 | 0.905 | 0.938 | 0.908 | 0.704 | 0.552 | 26.8 | 70.6 | 8.0 | 16.5 |
| YOLOv11n | 0.906 | 0.898 | 0.941 | 0.902 | 0.715 | 0.568 | 6.4 | 94.8 | 2.6 | 5.8 |
| RT-DETR | 0.905 | 0.908 | 0.935 | 0.906 | 0.695 | 0.548 | 57.0 | 68.7 | 20.0 | 75.0 |
| YOLO-world | 0.908 | 0.915 | 0.940 | 0.911 | 0.712 | 0.560 | 32.2 | 75.6 | 12.5 | 25.6 |
| **YOLO-GHS (Ours)** | **0.931** | **0.922** | **0.968** | **0.926** | **0.758** | **0.642** | **5.4** | **108.5** | **2.5** | **5.4** |

---

## 🚀 Quick Usage Guide

To reproduce our results or integrate YOLO-GHS into your pipeline:

**1. Inference / Evaluation**
You can directly use the provided `best.pt` with standard PyTorch/Ultralytics inference scripts to evaluate the model on your dataset:
```python
from ultralytics import YOLO

# Load our ultra-lightweight pre-trained model (5.4 MB)
model = YOLO('best.pt')

# Run inference on an image
results = model('path/to/your/image.jpg')
