# Phase 3: Explainability & Attribution Audit Environment

**Project:** Brain Atlas / Intracranial Aneurysm Research  
**Phase:** 3 — Explainability & Attribution Auditing  
**Date:** September 12, 2026  

---

## 1. Hardware & System Configuration

| Component | Specification |
| :--- | :--- |
| **Operating System** | Windows 11 Home / Professional (Build 26100) |
| **Host CPU** | Intel / AMD x86_64 Multi-Core Processor |
| **Host System RAM** | 32.0 GB Physical RAM |
| **GPU** | **NVIDIA GeForce RTX 5050 Laptop GPU** |
| **GPU Compute Architecture** | Ada Lovelace / Blackwell (`sm_120`) |
| **Dedicated VRAM** | **8.55 GB (8,546,484,224 bytes)** |
| **CUDA Driver / Runtime** | CUDA 13.0 / PyTorch CUDA 12.8 compatible |

---

## 2. Software & Deep Learning Frameworks

| Software Package | Version | Verification / Source |
| :--- | :--- | :--- |
| **Python** | 3.13.1 (64-bit) | Local Python distribution |
| **PyTorch** | `2.14.0+cu130` | GPU-accelerated build |
| **TorchVision** | `0.19.0` | Compatible release |
| **SimpleITK** | `2.4.0` | NIfTI image reading |
| **Nibabel** | `5.3.2` | Ground-truth mask extraction |
| **NumPy** | `2.1.2` | Vectorized numerical operations |
| **SciPy** | `1.14.1` | Connected component analysis |
| **Pandas** | `2.2.3` | Metrics table synthesis |
| **Matplotlib** | `3.9.2` | Publication visualization generation |

---

## 3. Model & Checkpoint Integrity

| Property | Value | Integrity Verification |
| :--- | :--- | :--- |
| **Model Architecture Class** | `ResEncoderUNet_two_seg_with_cls_modality` | Frozen baseline model |
| **Total Parameter Count** | **109,359,299** | Exact parameter count match |
| **Checkpoint Path** | `scratch/checkpoints/Dataset660_26classes_resize224_4661/onlyMirror01_lr4e3_100epochs_ps224/fold_0/checkpoint_final.pth` | Official Fold 0 weights |
| **Checkpoint SHA256** | `E60B539D025A8ECCCF77D3FF1A45EF6888B5E059671CC8E24FFF572D520FA521` | Authenticated cryptographic hash |
| **Model State** | `model.eval()` | Frozen weights; zero dropout |
| **Random Seed** | 42 | Deterministic execution |

---

## 4. XAI Technical Execution Protocols

| Setting | Configuration | Scientific Rationale |
| :--- | :--- | :--- |
| **Target Layer (Grad-CAM)** | `model.conv_encoder_blocks[5]` | Final encoder bottleneck ($320 \times 7 \times 7 \times 7$) feeding classification heads |
| **Target Scalar** | Pre-sigmoid logit of `cls_head_list[0][0, 0]` | Avoids sigmoid gradient saturation ($\frac{\partial \sigma}{\partial z} \to 0$ when $z \gg 0$) |
| **Forward Pass Mode** | `only_forward_cls=True` | Bypasses 3D decoders to isolate classification pathway and prevent CUDA OOM |
| **Grad-CAM Normalization** | $\text{ReLU}$, min-max scaling to $[0, 1]$ | Standard non-negative attribution convention |
| **Attention Extraction** | Hook on `cls_head_list[0].pooling.cross_attention` | Direct extraction of $2 \times 343$ query attention weights |
| **Integrated Gradients Baseline** | Uniform normalized zero ($0.0$) input | Represents absence of vascular contrast/signal |
| **IG Integration Steps** | $m = 20$ Riemann summation steps | Balances numerical convergence with GPU memory limits |
| **Faithfulness Strategy** | Progressive Deletion & Insertion (Top 1%, 5%, 10%) | Evaluates predictive score degradation without label contamination |
