## 📌 A Note on the Codebase (Code Availability)
To ensure clarity and highlight our exact methodological contributions, this repository provides the **core innovative modules** rather than a redundant clone of the entire Ultralytics framework. 

We have isolated and open-sourced the specific files that contain our mathematical and structural innovations:
* `yolo-ghs.yaml`: The complete structural configuration of our GhostHGNetV2 backbone and head.
* `seam.py`: The exact PyTorch implementation of the `Detect_SEAM` and the `Channel Exp` ($W_{exp} = e^Z$) mathematical mechanisms.
* `yolo-ghs-best.pt`: The pre-trained weights for direct evaluation.

**How to use:** This "plug-and-play" approach allows researchers to easily integrate our `Detect_SEAM` module and YAML configuration into the standard [Ultralytics](https://github.com/ultralytics/ultralytics) ecosystem without navigating thousands of lines of unmodified boilerplate code.



# YOLO-GHS-Insulator-Detection
YOLO-GHS
# YOLO-GHS: Lightweight and Robust Insulator Defect Detection in Adverse Weather

> **Official PyTorch implementation of the paper:** "YOLO-GHS: A Lightweight Object Detection Model for Insulator Defects in Adverse Weather via Spatial-Channel Exponential Attention" *(Under Review)*

---

## 💡 Overview

Deploying deep learning models on Unmanned Aerial Vehicles (UAVs) for power line inspection faces two critical bottlenecks: **stringent hardware constraints** (limited RAM/Compute) and **severe physical occlusion** caused by adverse weather (dense fog and heavy snow). 

**YOLO-GHS** is an ultra-lightweight, application-driven architecture designed to solve these exact challenges. By introducing a hardware-friendly backbone (**GhostHGNetV2**) and a novel mathematical attention mechanism (**Detect_SEAM**), YOLO-GHS achieves state-of-the-art performance with an exceptionally low memory footprint.

### 🔥 Key Highlights
* **Ultra-Lightweight for Edge AI:** Achieves **96.8% mAP@0.5** with only **5.4G FLOPs**, **2.5M parameters**, and a tiny **5.6 MB** model size. It is inherently lighter than the YOLOv11n baseline, making it perfect for UAV edge deployment.
* **Feature-Level Exposure Compensation:** Introduces the novel **Detect_SEAM** head. Unlike standard attention that uses linear Sigmoid attenuation (which irreversibly erases weak features under snow/fog), our `Channel Exp` function mathematically guarantees feature preservation using exponential compensation ($W_{exp} = e^Z$). When confidence drops due to occlusion ($Z \to 0$), the weight intrinsically converges to one ($W_{exp} \to 1$), unconditionally preserving the residual features.
* **Robust in Extreme Weather:** Achieves a massive boost in small object detection ($AP_S$ = 64.2%) on the real-world IDID dataset under severe weather splits.

---

## 🏗️ Architecture

![YOLO-GHS Architecture](docs/image1.png) 
*(Note: Please upload your overall network diagram and Figure 5 to a `docs` folder and link them here)*

The framework consists of:
1. **GhostHGNetV2 Backbone:** Replaces standard convolutions with Ghost convolutions to systematically halve intrinsic feature map channels, stripping away 1.6G FLOPs.
2. **Detect_SEAM Module (Spatial-channel Exponential Attention Module):**
   - **Stage 1 (CSMM):** Multi-scale spatial-channel extraction with parallel patch embeddings.
   - **Stage 2 (Channel Exp):** Non-linear feature compensation for occluded objects.

---

## 📊 Performance Comparison

Comprehensive comparative experiments  (Real-world Dense Fog & Heavy Snow):

| Model | Params (M) | Size (MB) | FLOPs (G) | mAP@0.5 (%) | $AP_S$ (%) | FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLOv5s | 7.0 | 14.1 | 16.8 | 88.4 | 46.3 | 48.7 |
| YOLOv10s | 8.0 | 16.5 | 26.8 | 93.8 | 55.2 | 70.6 |
| RT-DETR | 20.0 | 75.0 | 57.0 | 93.5 | 54.8 | 68.7 |
| YOLOv11n (Baseline) | 2.6 | 5.8 | 6.4 | 94.1 | 56.8 | 94.8 |
| **YOLO-GHS (Ours)** | **2.5** | **5.6** | **5.4** | **96.8** | **64.2** | **108.5** |

---

## ⚙️ Installation & Setup

1. Clone this repository:
```bash
git clone [https://github.com/songxiaoyang/YOLO-GHS.git](https://github.com/songxiaoyang/YOLO-GHS.git)
cd YOLO-GHS
