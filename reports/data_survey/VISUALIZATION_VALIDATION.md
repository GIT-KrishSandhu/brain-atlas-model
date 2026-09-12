# Data Survey: Visualization Validation Report (Issue #1)

## Executive Summary

- **Issue #1**: In the initial visualization outputs for TopAneu cases (e.g. `topaneu_center1_mr_024`), the **vessel overlay** visibly rendered vascular anatomy, but the **location overlay** and **type overlay** appeared visually identical or near-identical to the raw medical scan.
- **Root Cause Verified**: **Slice Selection Disconnect (Focal vs. Diffuse Anatomy)**. The initial visualization pipeline in `scratch/step5_visualize.py` selected orthogonal slices using static mid-volume coordinates (`shape[0]//2`, `shape[1]//2`, `shape[2]//2`). While extensive cerebrovascular vessel trees span across the volume mid-plane (intersecting thousands of voxels at the mid-slices), focal intracranial aneurysms occupy localized 3D bounding boxes. For `topaneu_center1_mr_024`, the aneurysm exists strictly between axial slices $z=68$ and $z=78$ ($y \in [218, 276]$). At mid-slice $z=80$ and $y=374$, the slice extracted from the location and type masks was **100% empty (all zeros)**. Matplotlib masked out all zeros (`masked_where(mask == 0, mask)`), rendering zero colored pixels over the raw grayscale scan.
- **Underlying Annotation Integrity**: The underlying NIfTI masks and annotations are **100% valid, intact, non-empty, and spatially aligned**. No annotation data was modified or compromised.
- **Fix Applied**: Upgraded the visualization pipeline to detect the 3D bounding box and focal peak slice for the targeted lesion, enforcing strict spatial synchronization across all 4 visual streams (Raw Scan, Vessel Overlay, Location Overlay, Type Overlay) using nearest-neighbor interpolation and categorical color mapping.

---

## 1. Diagnostic Verification of NIfTI Mask Data

Prior to modifying code, NIfTI files were inspected directly for `topaneu_center1_mr_024`:

| Stream | File Path | Shape | Dtype | Orientation | Unique Values | Non-Zero Voxels | 3D Bounding Box $[x, y, z]$ | Voxel Spacing (mm) |
|---|---|---|---|---|---|---|---|---|
| **Raw Image** | `images/topaneu_center1_mr_024_0000.nii.gz` | $(516, 749, 160)$ | `float64` | LPS | 817 values $[0, 824]$ | 25,660,197 | $[0, 0, 0] \to [515, 748, 159]$ | $(0.251, 0.251, 0.700)$ |
| **Vessel Mask** | `vessel_masks/topaneu_center1_mr_024.nii.gz` | $(516, 749, 160)$ | `float64` | LPS | $23$ values $[0 \dots 22]$ | 187,419 | $[138, 93, 2] \to [383, 442, 148]$ | $(0.251, 0.251, 0.700)$ |
| **Location Mask** | `location_masks/topaneu_center1_mr_024.nii.gz` | $(516, 749, 160)$ | `float64` | LPS | $\{0, 23, 24, 25\}$ | 1,734 | $[224, 218, 68] \to [307, 276, 78]$ | $(0.251, 0.251, 0.700)$ |
| **Type Mask** | `type_masks/topaneu_center1_mr_024.nii.gz` | $(516, 749, 160)$ | `float64` | LPS | $\{0, 1, 3\}$ | 1,734 | $[224, 218, 68] \to [307, 276, 78]$ | $(0.251, 0.251, 0.700)$ |

### Verification Findings:
1. **Mask Presence**: Both location and type masks are non-empty, containing 1,734 positive voxels each.
2. **Label Semantics**:
   - Location mask encodes multi-aneurysm classes: `23` (L-MCA), `24` (L-ICA), `25` (R-ICA).
   - Type mask encodes morphological classes: `1` (Saccular), `3` (Other/Complex).
3. **Spatial Alignment**: All 4 NIfTI files share identical dimensions $(516, 749, 160)$, identical LPS orientation, and identical affine transformation matrices.

---

## 2. Root Cause Analysis: Naive Mid-Slice vs. Focal Peak Slice

The reason the raw image and location/type overlays were visually identical in the original outputs:

```
Case: topaneu_center1_mr_024
Volume Z-dimension = 160 slices. Mid-slice z = 160 // 2 = 80.
Volume Y-dimension = 749 slices. Mid-slice y = 749 // 2 = 374.

Aneurysm 3D Extent:
Z: [68, 78]   --> Slice z = 80 contains ZERO aneurysm voxels (Count = 0)
Y: [218, 276] --> Slice y = 374 contains ZERO aneurysm voxels (Count = 0)
X: [224, 307] --> Mid-slice x = 258 intersects some voxels, but Coronal & Axial were 100% blank!

Vessel 3D Extent:
Z: [2, 148]   --> Slice z = 80 contains 2,870 vessel voxels!
```

- **Vessel Overlay**: Diffuse cerebral vasculature spans almost the entire brain ($z \in [2, 148]$). At naive $z=80$, 2,870 vessel voxels were present and vividly colored.
- **Location/Type Overlay**: At naive $z=80$, non-zero voxel count was **0**. The masked array `np.ma.masked_where(mask == 0, mask)` had no unmasked pixels, leaving `ax.imshow` with nothing to render. The viewer saw the underlying grayscale raw scan with 0 overlay pixels.

---

## 3. Code Modifications Applied

### A. Focal Coordinate Determination (`scratch/step5_visualize.py` & `scratch/visualization_issue1_diagnostic.py`)
Replaced naive center slicing with focal peak detection along all three orthogonal planes ($X, Y, Z$):

```python
# Determine focal slice coords (peak lesion voxels if positive, else peak vessel or volume center)
if loc_data is not None and np.any(loc_data > 0):
    sx = int(np.argmax(np.count_nonzero(loc_data, axis=(1, 2))))
    sy = int(np.argmax(np.count_nonzero(loc_data, axis=(0, 2))))
    sz = int(np.argmax(np.count_nonzero(loc_data, axis=(0, 1))))
elif ves_data is not None and np.any(ves_data > 0):
    sx = int(np.argmax(np.count_nonzero(ves_data, axis=(1, 2))))
    sy = int(np.argmax(np.count_nonzero(ves_data, axis=(0, 2))))
    sz = int(np.argmax(np.count_nonzero(ves_data, axis=(0, 1))))
else:
    sx, sy, sz = img_data.shape[0]//2, img_data.shape[1]//2, img_data.shape[2]//2

slices = (sx, sy, sz)
```

### B. Consistent Slice Projection and Nearest-Neighbor Rendering
All 4 panels in the orthogonal figures and diagnostic 2x2 figures use the identical slice coordinates $(sx, sy, sz)$.
Nearest-neighbor interpolation is strictly enforced to preserve categorical label identities:

```python
ax.imshow(
    masked.T,
    cmap=mask_cmap,
    alpha=alpha,
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    vmin=1,
    vmax=max_val
)
```

### C. Diagnostic 2x2 Quad-Panel Figure Function
Added `save_diagnostic_2x2(...)` producing unified figures with:
1. Top-Left: Raw Medical Scan (Slice $z$)
2. Top-Right: Raw Scan + Vessel Overlay (`autumn`, $\alpha=0.45$)
3. Bottom-Left: Raw Scan + Location Overlay (`tab20`, $\alpha=0.60$)
4. Bottom-Right: Raw Scan + Type Overlay (`Set1`, $\alpha=0.60$)

---

## 4. Multi-Case Validation Results

Validated across positive single-lesion, positive multi-lesion, and negative control cases:

| Case ID | Category | Focal Slices $(x, y, z)$ | Location Labels | Non-Zero Voxels (Focal Slice) | Location Voxels at Mid-Z | Resulting Overlay Visibility |
|---|---|---|---|---|---|---|
| `topaneu_center1_mr_017` | Positive (Single) | $(163, 174, 147)$ | $[28]$ (L-A2) | Loc: 78, Ves: 1,353 | $z=124 \to \mathbf{0}$ | **Vividly Visible** (Focal $z=147$) |
| `topaneu_center1_mr_024` | Positive (Multi) | $(289, 226, 75)$ | $[23, 24, 25]$ | Loc: 253, Ves: 2,348 | $z=80 \to \mathbf{0}$ | **Vividly Visible** (Focal $z=75$) |
| `topaneu_center1_mr_028` | Positive (Single) | $(134, 162, 150)$ | $[32]$ (R-P2) | Loc: 147, Ves: 1,123 | $z=124 \to \mathbf{0}$ | **Vividly Visible** (Focal $z=150$) |
| `topaneu_center1_mr_001` | Negative Control | $(206, 158, 84)$ | None ($[]$) | Loc: 0, Ves: 2,058 | $z=80 \to 0$ | **Correctly Empty** (Clean negative) |

---

## 5. Artifacts and Verification Deliverables

1. **Diagnostic Script**: `scratch/visualization_issue1_diagnostic.py` (standalone, reproducible test script).
2. **Structured JSON Report**: `scratch/visualization_issue1_report.json` (contains full affine, spacing, shape, label distributions, and slice metrics).
3. **Regenerated Visualizations**: Saved to `reports/data_survey/visualizations/`:
   - `topaneu_center1_mr_024_diagnostic_2x2.png`
   - `topaneu_center1_mr_024_image.png`
   - `topaneu_center1_mr_024_vessel_overlay.png`
   - `topaneu_center1_mr_024_location_overlay.png`
   - `topaneu_center1_mr_024_type_overlay.png`
   - Additional positive cases (`017`, `028`, `033`) and negative cases (`001`, `005`).

---

## 6. Confirmation of Constraints

- **No Medical Annotations Modified**: All NIfTI files in `topaneu_release` were accessed read-only (`get_fdata()`).
- **No P3 Model Code Modified**: No architecture, checkpoints, or weights touched.
- **Vessel Regression**: Vessel masks remain visible and correctly rendered across all cases.
