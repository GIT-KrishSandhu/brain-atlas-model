# Data Availability & Dataset Inventory Audit
## Brain Atlas / Intracranial Aneurysm Research Project
**Audited Target:** Local storage vs. RSNA 2025 P3 Baseline Requirements  
**Date:** 2026-09-10  

---

## 1. Dataset Availability Matrix

| Dataset / Component | Present? | Location | Number of Cases | Format | Notes / Downstream Dependency |
|---|---|---|---:|---|---|
| **TopAneu-26 Scans** | ✅ **YES** | `D:\NLP_Project\topaneu_release\images\` | 415 real scans (+415 ghost `._` files) | 3D NIfTI (`.nii.gz`) | Primary available dataset. Spacing 0.25–0.625mm. |
| **TopAneu Vessel Masks** | ✅ **YES** | `D:\NLP_Project\topaneu_release\vessel_masks\` | 415 real masks | 3D NIfTI (`.nii.gz`) | 36 anatomical vessel classes + background. |
| **TopAneu Location Masks**| ✅ **YES** | `D:\NLP_Project\topaneu_release\location_masks\` | 415 real masks | 3D NIfTI (`.nii.gz`) | 52 aneurysm location classes + background. |
| **TopAneu Type Masks** | ✅ **YES** | `D:\NLP_Project\topaneu_release\type_masks\` | 415 real masks | 3D NIfTI (`.nii.gz`) | 3 aneurysm type classes (saccular, dissecting, fusiform). |
| **TopAneu Location JSONs**| ✅ **YES** | `D:\NLP_Project\topaneu_release\location_jsons\` | 415 real JSONs | `.json` metadata | 304 positive cases, 111 negative cases. |
| **RSNA 2025 Raw DICOM** | ❌ **NO** | Not on disk (hosted on Kaggle) | 0 locally (expected ~4,661 series) | DICOM series | Required for reproducing raw preprocessing pipeline. |
| **RSNA Localizers CSV** | ❌ **NO** | Not on disk (hosted on Kaggle) | 0 locally (`train_localizers.csv`) | `.csv` | Coordinates for aneurysm bounding box generation. |
| **Dataset180 (Stage 1)** | ❌ **NO** | Not on disk (hosted on Kaggle) | 0 locally (expected 504 sample cases) | nnU-Net 2D format | Required to train/reproduce Stage 1 2D ROI model. |
| **Dataset660 (Stage 2)** | ❌ **NO** | Not on disk (hosted on Kaggle) | 0 locally (expected 4,661 cases) | nnXNet 3D preprocessed | Required to train P3 Stage 2 model as-is. |
| **26-Class Seg Masks** | ❌ **NO** | Not on disk (hosted on HuggingFace: `spc819/rsna2025-aneurysm-26class-seg`)| 0 locally | NIfTI / NumPy | Corrected masks used to construct Dataset660. |
| **`train_augmented.csv`**| ❌ **NO** | Not on disk (hardcoded path in dataloader) | 0 locally | `.csv` | Required by `nnXNetDataLoader3DWithGlobalCls`. |
| **Sampling Weight CSV** | ❌ **NO** | Not on disk (hardcoded path in dataloader) | 0 locally | `.csv` | Cached inverse class frequency sampling table. |
| **Stage 1 Pretrained Model**| ❌ **NO** | Not on disk (hosted on Kaggle) | 0 locally | `.pth` checkpoint | Stage 1 2D vessel segmentation weights. |
| **Stage 2 Pretrained Model**| ❌ **NO** | Not on disk (hosted on Kaggle) | 0 locally | `.pth` checkpoint | Stage 2 Fold 0 / Fold 1 weights. |

---

## 2. Checkpoint Availability Detail

| Model | Expected Checkpoint Path | Found Locally? | Remote Source | Compatible with Architecture? |
|---|---|---|---|---|
| **Stage 1 ROI Model** | `dataset180_2d_vessel_box_seg_stable/fold_0/checkpoint_final.pth` | ❌ **NO** | `https://www.kaggle.com/models/pengchengshi/dataset180_2d_vessel_box_seg_stable` | Yes (nnUNet 2D) |
| **Stage 2 Model (Fold 0)**| `dataset660_26classes_resize224_4661/fold_0/checkpoint_final.pth` | ❌ **NO** | `https://www.kaggle.com/models/pengchengshi/dataset660_26classes_resize224_4661` | Yes (`ResEncoderUNet_two_seg_with_cls_modality`) |
| **Stage 2 Model (Fold 1)**| `dataset660_26classes_resize224_4661/fold_1/checkpoint_final.pth` | ❌ **NO** | `https://www.kaggle.com/models/pengchengshi/dataset660_26classes_resize224_4661` | Yes (`ResEncoderUNet_two_seg_with_cls_modality`) |
| **Stage 2 Best Checkpoint**| `dataset660_26classes_resize224_4661/fold_0/checkpoint_best.pth` | ❌ **NO** | Hosted on Kaggle | Yes (`ResEncoderUNet_two_seg_with_cls_modality`) |

---

## 3. Conclusions on Data Readiness

1. **Full RSNA 2025 Training Pipeline:** Currently **BLOCKED** due to missing `Dataset660`, missing `train_augmented.csv`, and missing raw competition DICOMs.
2. **Pretrained Baseline Inference:** Currently **BLOCKED** due to missing Kaggle checkpoints.
3. **Pipeline & Architectural Validation:** Fully **FEASIBLE and VERIFIED** using the 415 high-quality multi-modal 3D scans in `topaneu_release` (304 positive, 111 negative). All forward-pass, backward-pass, and tiny-subset overfit experiments succeeded using TopAneu scans.
