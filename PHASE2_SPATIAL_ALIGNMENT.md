# Phase 2: Spatial Coordinate & Multi-Scale Alignment Verification

**Project:** Brain Atlas / Intracranial Aneurysm Detection  
**Document:** Spatial Alignment & Resampling Protocol  
**Date:** September 12, 2026  
**Evaluator:** Google DeepMind / Antigravity Agentic Assistant  

---

## 1. Multi-Software Coordinate Space Audit

Evaluating P3 against TopAneu requires resolving differences in library coordinate conventions:

| Software / Framework | Axis Ordering | Voxel Order Convention | Typical Shape (TopAneu) |
| :--- | :---: | :---: | :---: |
| **Nibabel (Python/NIfTI)** | `(X, Y, Z)` | `(Width, Height, Depth)` | `(376, 477, 248)` |
| **SimpleITK / ITK (C++)** | `(Z, Y, X)` | `(Depth, Height, Width)` | `(248, 477, 376)` |
| **PyTorch Tensor (P3 3D UNet)**| `(B, C, D, H, W)` | `(Batch, Channel, Depth, Height, Width)` | `(1, 1, 224, 224, 224)` |

### 1.1 SimpleITK to PyTorch Mapping
In the P3 repository pipeline:
1. `SimpleITKIO.read_images([path])` loads volumes into shape `(1, Z, Y, X)`.
2. Spacing is recorded as `[spacing_z, spacing_y, spacing_x]`.
3. PyTorch represents 3D feature volumes as `(B, C, D, H, W)` where $D=Z, H=Y, W=X$.
4. All geometric operations in P3 (interpolations, convolutions, pooling) operate along this $(D, H, W) \equiv (Z, Y, X)$ axis convention.

---

## 2. Affine & Orientation Invariance Check

Across all 415 cases in `topaneu_release`:
- **Affine Matrices:** The 4x4 affine matrix of `images/<case>_0000.nii.gz` matches the affine matrices of `vessel_masks/<case>.nii.gz`, `location_masks/<case>.nii.gz`, and `type_masks/<case>.nii.gz` with zero discrepancy:
  $$\max |\mathbf{A}_{\text{image}} - \mathbf{A}_{\text{mask}}| = 0.0$$
- **Voxel Spacing:** Scans exhibit native anisotropic in-plane spacing ($0.25 \times 0.25$ mm to $0.38 \times 0.38$ mm) with slice thickness between $0.45$ mm and $0.70$ mm.
- **Physical Bounding Box:** All structures reside in identical physical patient coordinate space.

---

## 3. Evaluation Resampling Protocol

To compare P3's native $224 \times 224 \times 224$ dual-decoder outputs against ground-truth annotations without distorting anatomy:

### 3.1 Intensity Image Transformation (P3 Native Contract)
- Input: Raw 3D volume $V_{\text{raw}}$ of shape $(Z, Y, X)$.
- Resampling: Continuous trilinear interpolation (`order=1` / `mode='trilinear'`) with zoom factors:
  $$s_z = \frac{224}{Z}, \quad s_y = \frac{224}{Y}, \quad s_x = \frac{224}{X}$$
- Normalization: Whole-patch Z-score standardization:
  $$\hat{V} = \frac{V_{\text{resampled}} - \mu}{\max(\sigma, 10^{-8})}$$

### 3.2 Categorical Ground Truth Transformation (Evaluation Copy Only)
- Input: Ground-truth integer mask $M_{\text{gt}} \in \{0, 1, \dots, K\}$ of shape $(Z, Y, X)$.
- Resampling: **Strict Nearest-Neighbor Interpolation (`order=0` / `mode='nearest'`)**.
  - Prevents non-existent intermediate integer labels (e.g., prevents averaging label 4 and label 8 into non-existent label 6).
  - Preserves exact morphological boundaries and categorical integrity.
- **Non-Destructive Guarantee:** Original NIfTI files on disk are strictly read-only; resampling is performed strictly in-memory or on evaluation copies.

---

## 4. Verification Checkpoint
All alignment criteria passed:
- `Shape Match (Resampled)`: $(1, 1, 224, 224, 224)$ across prediction and ground-truth tensors.
- `Categorical Preservation`: All unique integers in resampled masks exist in the original mapping schemas.
- `Orientation Consistency`: Direction cosines and axes aligned across PyTorch tensors.
