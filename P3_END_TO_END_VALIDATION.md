# P3 End-to-End Validation: Stage-1 Vessel ROI to Stage-2 Classification

**Project:** Brain Atlas / Intracranial Aneurysm Detection  
**Evaluation Scope:** Complete End-to-End Inference Pipeline on TopAneu Test Cases  
**Date:** September 12, 2026  
**Evaluator:** Google DeepMind / Antigravity Agentic Assistant  

---

## 1. Objective & Pipeline Overview

The goal of this experiment is to validate whether the original two-stage published P3 inference pipeline operates end-to-end on local TopAneu data:

```
Raw 3D Scan (NIfTI / SimpleITK)
       ↓
Stage 1: PlainConvUNet 2D (Tri-axial candidate slices Z, Y, X)
       ↓
Vessel Mask Prediction & Per-axis Coordinate Bounding
       ↓
Bounding-Box Fusion (Coordinate Arithmetic Mean)
       ↓
3D ROI Cropping
       ↓
224³ Preprocessing (Z-Score Norm + Trilinear Resampling)
       ↓
Stage 2: Pretrained ResEncoderUNet_two_seg_with_cls_modality
       ↓
Aneurysm Presence + Location + Modality Predictions
```

> [!IMPORTANT]
> This is a **Baseline Validation Task**, not a model improvement task. No model weights, architectures, losses, or thresholds have been modified.

---

## 2. Test Cohort & Execution Summary

Evaluation was conducted on four representative TopAneu cases:
- **Positive Cases:** `topaneu_center1_mr_017`, `topaneu_center1_mr_024`, `topaneu_center1_mr_028`
- **Negative Case:** `topaneu_center1_mr_001`

### 2.1 Complete Execution Metrics Matrix

| Metric | `topaneu_center1_mr_017` | `topaneu_center1_mr_024` | `topaneu_center1_mr_028` | `topaneu_center1_mr_001` |
| :--- | :--- | :--- | :--- | :--- |
| **Case Category** | **Positive** (Aneurysm) | **Positive** (Aneurysm) | **Positive** (Aneurysm) | **Negative** (Normal) |
| **Imaging Modality** | MRA (TOF) | MRA (TOF) | MRA (TOF) | MRA (TOF) |
| **Input Shape $(C, Z, Y, X)$** | `(1, 248, 477, 376)` | `(1, 160, 749, 516)` | `(1, 248, 438, 368)` | `(1, 264, 426, 352)` |
| **Voxel Spacing (mm)** | `[0.4500, 0.3819, 0.3819]` | `[0.7000, 0.2511, 0.2511]` | `[0.4500, 0.3819, 0.3819]` | `[0.4500, 0.3819, 0.3819]` |
| **Axial Candidates (Z)** | `[124, 62, 186]` | `[80, 40, 120]` | `[124, 62, 186]` | `[132, 66, 198]` |
| **Coronal Candidates (Y)** | `[238, 119, 357]` | `[374, 187, 561]` | `[219, 109, 328]` | `[213, 106, 319]` |
| **Sagittal Candidates (X)**| `[188, 94, 282]` | `[258, 129, 387]` | `[184, 92, 276]` | `[176, 88, 264]` |
| **Stage 1 Fused BBox** | `(2, 248, 47, 355, 46, 333)` | `(0, 160, 58, 625, 49, 475)` | `(0, 248, 54, 320, 45, 325)` | `(0, 264, 56, 313, 40, 310)` |
| **ROI Dims Before Resize** | `246 × 308 × 287` | `160 × 567 × 426` | `248 × 266 × 280` | `264 × 257 × 270` |
| **ROI Volume Reduction** | **52.2% of raw volume** | **62.9% of raw volume** | **46.8% of raw volume** | **45.9% of raw volume** |
| **ROI Dims After Resize** | `224 × 224 × 224` | `224 × 224 × 224` | `224 × 224 × 224` | `224 × 224 × 224` |
| **Fallback Triggered?** | **No** (Valid ROI) | **No** (Valid ROI) | **No** (Valid ROI) | **No** (Valid ROI) |
| **Stage-1 Time (s)** | 3.418 s | 6.326 s | 2.028 s | 0.282 s |
| **Stage-1 Peak VRAM** | 1,992.6 MB (~1.95 GB) | 3,497.5 MB (~3.42 GB) | 2,980.6 MB (~2.91 GB) | 2,330.0 MB (~2.28 GB) |
| **GT Aneurysm Voxels** | 576 voxels | 1,734 voxels | 327 voxels | 0 voxels (N/A) |
| **GT Aneurysm BBox** | `Z[143:153], Y[168:178], X[158:169]` | `Z[68:79], Y[218:277], X[224:308]` | `Z[148:154], Y[156:166], X[131:141]` | None |
| **Contained GT Voxels** | **576 / 576 (100.0%)** | **1,734 / 1,734 (100.0%)** | **327 / 327 (100.0%)** | N/A |
| **Aneurysm Contained?** | ✅ **YES (100%)** | ✅ **YES (100%)** | ✅ **YES (100%)** | ✅ **N/A** |
| **Stage-2 Inference Time**| 8.077 s | 3.610 s | 0.758 s | 0.428 s |
| **Stage-2 Peak VRAM** | 5,695.8 MB (~5.56 GB) | 4,226.3 MB (~4.13 GB) | 4,149.3 MB (~4.05 GB) | 4,148.8 MB (~4.05 GB) |
| **End-to-End Latency** | 11.495 s | 9.936 s | 2.786 s | 0.710 s |

---

## 3. Downstream Predictions: P3 Two-Stage vs Whole-Volume Baseline

Below is the head-to-head comparison between:
- **Baseline A:** Stage 2 evaluated on direct full-scan isotropic resampling (current baseline).
- **Pipeline B:** Stage 2 evaluated on the Stage-1 tri-axial vessel ROI crop.

| Case ID | GT Presence | Baseline A Presence Prob | **Pipeline B Presence Prob** | Baseline A Predicted Loc | **Pipeline B Predicted Loc** | Predicted Modality |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `topaneu_center1_mr_017` | **1** (Positive) | 0.2406 (FN @ 0.5) | **0.8765 (TP @ 0.5)** | Right Supraclinoid ICA | **Right Supraclinoid ICA** | MRA |
| `topaneu_center1_mr_024` | **1** (Positive) | 0.3606 (FN @ 0.5) | **0.0868 (FN @ 0.5)** | Right MCA | **Right Supraclinoid ICA** | MRA |
| `topaneu_center1_mr_028` | **1** (Positive) | 0.4700 (FN @ 0.5) | **0.9634 (TP @ 0.5)** | Other Posterior Circ | **Right Pcom** | MRA |
| `topaneu_center1_mr_001` | **0** (Negative) | 0.0850 (TN @ 0.5) | **0.0320 (TN @ 0.5)** | Right Supraclinoid ICA | **Other Posterior Circ** | MRA |

### 3.1 Observations
1. **Presence Probability Sensitivity:**
   - On `topaneu_center1_mr_017`, presence probability increased dramatically from **0.2406** (false negative under full volume) to **0.8765** (strong true positive).
   - On `topaneu_center1_mr_028`, presence probability increased from **0.4700** to **0.9634** (strong true positive), and the anatomical location prediction sharpened to `Right Pcom` with 92.7% posterior probability.
   - On `topaneu_center1_mr_001` (Negative), background false-positive noise dropped further from **0.0850** down to **0.0320** (strong true negative).
   - On `topaneu_center1_mr_024`, presence probability dropped from 0.3606 to 0.0868. Notably, case 024 has a substantially higher resolution ($749 \times 516$) and non-standard spacing ($0.7 \times 0.25 \times 0.25$ mm), resulting in a broader ROI crop ($160 \times 567 \times 426$).
2. **Modality Invariance:**
   - The Stage-2 modality classification head unanimously predicted **MRA** (100% correct across all cases).

---

## 4. Failure & Containment Analysis

### 4.1 Positive Case Containment Rate
A critical safety concern for two-stage detection pipelines is whether Stage 1 inadvertently crops out the pathology:

$$\text{ROI Containment Rate} = \frac{\text{Positive cases whose aneurysm mask intersects Stage-1 ROI}}{\text{Total positive cases tested}}$$

$$\text{ROI Containment Rate} = \frac{3}{3} = \mathbf{100.0\%}$$

Every single ground-truth aneurysm voxel across all three positive cases (576 voxels in case 017, 1,734 voxels in case 024, and 327 voxels in case 028) was **100% contained** within the predicted Stage-1 bounding box.

### 4.2 Failure Mode Audit
- **Stage-1 Prediction Failures:** **0** (All cases generated valid vessel predictions).
- **Empty Vessel Masks:** None. Every sampled orthogonal plane produced valid vessel segmentations.
- **Invalid Bounding Boxes:** None ($z_{\min} < z_{\max}$, $y_{\min} < y_{\max}$, $x_{\min} < x_{\max}$ held strictly for all cases).
- **Fallback Trigger Rate:** **0% (0 / 4)**. No case reverted to full-volume fallback.
- **ROI Truncation:** Zero truncation of intracranial vascular territories was observed.
- **VRAM Constraints:** Peak memory during Stage 1 was 3.50 GB; peak memory during Stage 2 was 5.70 GB. Both fit comfortably within the 8 GB RTX 5050 Laptop GPU memory budget.

---

## 5. Diagnostic Visualizations

Diagnostic figures were generated programmatically and saved to `scratch/stage1_roi_visualizations/`:

1. **`topaneu_center1_mr_017`:**
   - Raw Scan vs Stage-1 Candidate Predictions vs BBox Overlay vs $224^3$ Input:  
     `scratch/stage1_roi_visualizations/topaneu_center1_mr_017_stage1_roi_diagnostic.png`
2. **`topaneu_center1_mr_024`:**
   - Diagnostic Visualization:  
     `scratch/stage1_roi_visualizations/topaneu_center1_mr_024_stage1_roi_diagnostic.png`
3. **`topaneu_center1_mr_028`:**
   - Diagnostic Visualization:  
     `scratch/stage1_roi_visualizations/topaneu_center1_mr_028_stage1_roi_diagnostic.png`
4. **`topaneu_center1_mr_001` (Negative):**
   - Diagnostic Visualization:  
     `scratch/stage1_roi_visualizations/topaneu_center1_mr_001_stage1_roi_diagnostic.png`

---

## 6. Important Data Limitation & Domain Shift

TopAneu is an external cohort with distinct slice thickness, scan protocols, and field-of-view characteristics compared to the RSNA 2025 Dataset180 and Dataset660 training cohorts:
- TopAneu MRA scans possess high anisotropic in-plane resolution (up to $0.25 \times 0.25$ mm) with variable slice counts (160–264 slices).
- While Stage 1 successfully extracted intracranial vascular ROIs with 100% aneurysm containment on all tested positive cases, this experiment establishes **pipeline interoperability and functional execution only**, and does not constitute a full reproduction of original P3 competition metrics on TopAneu.
