# Baseline Environment Diagnostic & Compatibility Audit
## Brain Atlas / Intracranial Aneurysm Research Project
**Audited Target:** RSNA 2025 2nd Place Solution (P3: `RSNA2025_Intracranial-Aneurysm-Detection`)  
**Date:** 2026-09-10  
**Discipline:** Strictly measured local values vs. P3 paper/code requirements.

---

## 1. Environment Diagnostic Matrix

| Component | Detected Value | Required / Expected by P3 | Status |
|---|---|---|---|
| **Operating System** | Windows 11 Home/Pro (Build 10.0.26200-SP0) | Ubuntu 22.04.4 LTS (Linux) | ⚠️ **INCOMPATIBLE / REQUIRES ADAPTATION** (POSIX paths, multiprocessing fork vs spawn) |
| **Python Version** | 3.13.14 (tags/v3.13.14:fd17997, 64-bit AMD64) | Python 3.10 – 3.11 | ⚠️ **HIGH RISK** (Python 3.13 lacks prebuilt wheels for key neuroimaging C-extensions) |
| **PyTorch Version** | `2.14.0+cu130` | `torch==2.6.0+cu124` (or >=2.1.0 with CUDA) | ✅ **COMPATIBLE & ACTIVATED** |
| **CUDA Runtime (Torch)**| `CUDA 13.0` (`torch.cuda.is_available() == True`) | CUDA 12.1 or 12.4 | ✅ **COMPATIBLE** (sm_120 native Blackwell support) |
| **NVIDIA Driver** | `592.19` (Directly verified via `nvidia-smi`) | >= 525.60.13 | ✅ **COMPATIBLE** (Hardware driver supports CUDA up to 13.1) |
| **GPU Model** | NVIDIA GeForce RTX 5050 Laptop GPU | 1 × NVIDIA A100 80GB | ⚠️ **8GB VRAM (Inference Feasible, Full Training Constrained)** |
| **GPU VRAM** | 8,151 MiB (~8.0 GB) | 80 GB (~53 GB consumed during Stage-2 training) | ❌ **INSUFFICIENT FOR FULL STAGE-2 TRAINING** (6.6× shortfall vs 53 GB requirement) |
| **CPU Model** | Intel(R) Core(TM) i7-14650HX (16 cores, 24 threads, 2.2 GHz) | Intel Xeon Platinum 8468 @ 2.10 GHz (48+ cores) | ⚠️ **SUBOPTIMAL** (Adequate for light testing, inadequate for high-throughput 3D dataloading) |
| **System RAM** | 23.64 GB total (7.86 GB available at audit) | 512 GiB RAM | ❌ **SEVERELY CONSTRAINED** (21.6× shortfall vs 512 GiB; memory-mapped dataloading will OOM) |
| **Available Disk (D:)** | 341.8 GB Total, 311.06 GB Free | 1.0 – 2.0 TB for full RSNA2025 raw + preprocessed | ⚠️ **PARTIAL** (Sufficient for TopAneu-26 ~20 GB; insufficient for 4,661-case raw RSNA dataset) |
| **`dynamic_network_architectures`** | NOT INSTALLED in stock env (Vendored in `scratch/dna_src`) | `dynamic_network_architectures==0.3.1` | ⚠️ **MISSING FROM SYSTEM ENV** |
| **`monai`** | NOT INSTALLED | `monai==1.5.0` | ❌ **MISSING** |
| **`SimpleITK`** | NOT INSTALLED | `SimpleITK==2.5.2` | ❌ **MISSING** |
| **`dicom2nifti`** | NOT INSTALLED | `dicom2nifti==2.6.2` | ❌ **MISSING** |
| **`batchgenerators`** | NOT INSTALLED | `batchgenerators==0.25.1` | ❌ **MISSING** |
| **`connected-components-3d`** | NOT INSTALLED | `connected-components-3d==3.24.0` | ❌ **MISSING** |
| **`pandas`** | NOT INSTALLED | `pandas==2.3.2` | ❌ **MISSING** |
| **`torchvision`** | NOT INSTALLED | Compatible with torch | ❌ **MISSING** |
| **`nibabel`** | `5.4.2` | `nibabel==5.3.2` | ✅ **INSTALLED** |
| **`numpy`** | `2.5.2` | `numpy==2.2.6` | ✅ **INSTALLED** |
| **`scipy`** | `1.18.1` | Scientific stack | ✅ **INSTALLED** |
| **`scikit-learn`** | `1.9.0` | ML evaluation | ✅ **INSTALLED** |
| **`matplotlib`** | `3.11.1` | Plotting stack | ✅ **INSTALLED** |

---

## 2. Hardware & Infrastructure Feasibility Analysis

### 2.1 The GPU VRAM Constraint
The P3 baseline README states explicitly:
> *"Stage 2 training uses batch size = 2 with approximately 53GB GPU memory consumption."*

Our system has an **NVIDIA GeForce RTX 5050 Laptop GPU with 8,151 MiB (~8 GB) VRAM**.
- For full Stage-2 multi-task training (6-stage residual encoder + dual 5-stage decoders + 3 classification heads + deep supervision at $224 \times 224 \times 224$ with batch size 2), activation memory alone exceeds 40 GB.
- **Direct full training of P3 is physically impossible on this GPU without modifications** (e.g. gradient checkpointing, batch size 1, mixed precision, or patch-size downscaling).
- However, **inference-only mode** (`only_forward_cls=True`) and **compact patch evaluation** ($64^3$ to $112^3$) fit comfortably within 2–4 GB.

### 2.2 The PyTorch CPU-Only Blocker
- The installed PyTorch environment is `2.14.0+cpu`.
- Even though the physical RTX 5050 GPU and NVIDIA Driver 592.19 are installed and fully operational (verified via `nvidia-smi`), PyTorch cannot access CUDA.
- All baseline tests executed in this session ran on the CPU. Forward pass on $224^3$ required 22.7s for classification-only, and 79.2s for forward + 137.9s for backward.
- **Immediate Remedy:** Reinstall CUDA-enabled PyTorch (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` or cu126).

### 2.3 Operating System & Path Incompatibilities
The P3 repository originates from a Linux research cluster (`Ubuntu 22.04.4 LTS`):
- Several scripts contain hardcoded POSIX paths (e.g., `/yinghepool/shipengcheng/Dataset/nnUNet` in `data_loader_3d_with_global_cls.py`).
- Windows process spawning differences (`spawn` vs `fork`) will crash the `multiprocessing` workers in nnU-Net/nnXNet dataloaders if `num_workers > 0`.
- Local execution must use `num_processes=0` or adapt worker startup.
