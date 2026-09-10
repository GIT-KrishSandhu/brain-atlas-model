# Environment Audit
## Computational Environment -- Brain Atlas / TopAneu-26

**Generated:** 2026-09-08

## Python
- **Version:** 3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026) [MSC v.1944 64 bit (AMD64)]
- **Executable:** C:\Users\DIYA\AppData\Local\Programs\Python\Python313\python.exe
- **Platform:** Windows-11-10.0.26200-SP0

## Python Package Inventory

| Package | Status |
|---|---|
| `torch` | 2.14.0+cpu |
| `torchvision` | NOT INSTALLED |
| `nibabel` | 5.4.2 |
| `numpy` | 2.5.2 |
| `scipy` | 1.18.1 |
| `matplotlib` | 3.11.1 |
| `pandas` | NOT INSTALLED |
| `scikit-learn` | 1.9.0 |
| `monai` | NOT INSTALLED |
| `SimpleITK` | NOT INSTALLED |
| `nnunetv2` | NOT INSTALLED |
| `dicom2nifti` | NOT INSTALLED |

## GPU / CUDA

- **torch.cuda.is_available():** `False`
- **PyTorch version:** `2.14.0+cpu`
- **CUDA version (torch):** `None`
- **nvidia-smi:** Not found on PATH
- **CUDA status:** UNAVAILABLE -- CPU-only PyTorch installed

> CRITICAL: GPU training is not possible until CUDA-enabled PyTorch is installed.
> Recommended: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

## System RAM
- Total: 23.64 GB
- Available (at audit): 9.21 GB
- Used: 14.42 GB

## Storage
- D:\ Total: 341.8 GB
- D:\ Used: 30.7 GB
- D:\ Free: 311.1 GB

## Required Package Installations

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pandas monai SimpleITK nnunetv2 dicom2nifti
```
