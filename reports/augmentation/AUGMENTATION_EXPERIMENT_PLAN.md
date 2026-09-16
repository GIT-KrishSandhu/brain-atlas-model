# Data Augmentation Phase: Experiment Plan & Protocol Specification

**Project:** Brain Atlas / Intracranial Aneurysm Research  
**Investigation Scope:** Controlled Data Augmentation of Fine-Tuned P3 on TopAneu-26  
**Date:** September 16, 2026  
**Evaluator:** Google DeepMind / Antigravity Agentic Assistant  
**Evidence Discipline:** `[MEASURED]`, `[METADATA-DERIVED]`, `[CODEBASE-DERIVED]`, `[LITERATURE-DERIVED]`, `[INFERRED]`, `[REQUIRES VALIDATION]`

---

## 1. Research Question & Hypothesis

### 1.1 Formal Research Question
$$\text{"Does controlled augmentation of the training distribution improve the performance and robustness of the fine-tuned P3 model on TopAneu-26?"}$$

### 1.2 Scientific Hypotheses
- **Null Hypothesis ($H_0$):** Stochastic perturbation of the training distribution via controlled 3D spatial and intensity augmentations produces no statistically or clinically meaningful improvement in discrimination (ROC-AUC $\Delta \le 0.005$) or probabilistic calibration (Brier score $\Delta \ge -0.005$) on the locked TopAneu-26 test set.
- **Alternative Hypothesis ($H_1$):** Controlled data augmentation improves model generalization and feature invariance, reducing false positive/negative rates and increasing ROC-AUC/PR-AUC on the locked test set without modifying the underlying P3 architecture.

---

## 2. Experimental Structure: A0 Control vs. AUG Condition

To maintain strict scientific control, exactly two experimental conditions are compared:

### 2.1 Condition A0 (Frozen Control)
- **Model:** Native `P3Architecture` (109,359,299 parameters). `[CODEBASE-DERIVED]`
- **Weights:** Fine-tuned baseline checkpoint (`P4-P3-001/best_checkpoint.pth`). `[CODEBASE-DERIVED]`
- **Training Augmentation:** **0.0% (Identity transform)**.
- **Training Epochs:** 3 epochs, AdamW optimizer ($\text{lr} = 10^{-4}, \text{wd} = 10^{-4}$), CosineAnnealingLR ($\eta_{\min} = 5 \times 10^{-6}$), batch size 1 with gradient accumulation steps 8, AMP FP16. `[CODEBASE-DERIVED]`
- **Observed Locked Test Performance ($N=62$):**
  - ROC-AUC: 0.8928 `[MEASURED]`
  - PR-AUC: 0.9626 `[MEASURED]`
  - Accuracy: 77.42% `[MEASURED]`
  - Sensitivity: 95.56% `[MEASURED]`
  - Specificity: 29.41% `[MEASURED]`
  - F1-Score: 0.8600 `[MEASURED]`
  - Brier Score: 0.1294 `[MEASURED]`

### 2.2 Condition AUG (Augmentation Treatment)
- **Experiment ID:** `AUG-P3-001`
- **Model:** Identical native `P3Architecture` (109,359,299 parameters). `[CODEBASE-DERIVED]`
- **Initialization:** Identical official P3 checkpoint (`Dataset660_26classes_resize224_4661/.../checkpoint_final.pth`). `[CODEBASE-DERIVED]`
- **Optimization:** Exactly identical hyperparameters (AdamW, lr=1e-4, wd=1e-4, CosineAnnealingLR, 3 epochs, batch size 1, grad accum 8, AMP FP16, pos_weight=1.25).
- **Split:** Exactly identical 290 training scans from `experiments/splits/topaneu_v1.csv`.
- **Training Augmentation:** Active stochastic augmentation pipeline applied on-the-fly to training cases.
- **Validation:** Identical unaugmented validation partition ($N=63$) for checkpoint selection.
- **Evaluation:** Identical unaugmented locked test partition ($N=62$) evaluated only after training is completed.

---

## 3. Approved Augmentations & Parameterization

Only the following six augmentation techniques are approved for implementation:

### 3.1 Technique A: Small 3D Rotation
- **Parameter Range:** Rotation angles $\theta_x, \theta_y, \theta_z \sim \mathcal{U}(-10^\circ, +10^\circ)$. `[INFERRED]`
- **Probability:** $p = 0.40$.
- **Rationale:** Simulates physical patient head tilt within the scanner bore. The inferred candidate range from `DATA_SURVEY.md` was $\pm 15^\circ$; $\pm 10^\circ$ is selected as a conservative bound to prevent clipping the cerebral convexity against volume boundaries. `[INFERRED]`

### 3.2 Technique B: Small 3D Translation
- **Parameter Range:** Voxel displacements $\Delta x, \Delta y, \Delta z \sim \mathcal{U}(-8, +8)$ voxels ($\sim \pm 3.6\%$ of the 224-voxel FOV). `[INFERRED]`
- **Probability:** $p = 0.40$.
- **Rationale:** Simulates scanner field-of-view (FOV) isocenter alignment variation across clinical sites. Displacements are conservative to ensure the Circle of Willis remains well within the active receptive field. `[INFERRED]`

### 3.3 Technique C: Mild Contrast Variation
- **Parameter Range:** Contrast factor $c \sim \mathcal{U}(0.90, 1.10)$. `[INFERRED]`
- **Probability:** $p = 0.30$.
- **Formula:** $I_{\text{aug}} = I \times c$.
- **Rationale:** Perturbs vascular-to-parenchymal contrast. Because the volume is Z-score normalized ($\mu \approx 0$), this scales variance without inducing mean shift. `[INFERRED]`

### 3.4 Technique D: Mild Gamma Transformation
- **Parameter Range:** Exponent $\gamma \sim \mathcal{U}(0.90, 1.10)$. `[INFERRED]`
- **Probability:** $p = 0.30$.
- **Formula:** Dynamically min-max normalized: $I_{\text{scaled}} = \frac{I - I_{\min}}{I_{\max} - I_{\min} + 10^{-8}}$, $I_{\gamma} = (I_{\text{scaled}})^\gamma$, $I_{\text{aug}} = I_{\gamma} \times (I_{\max} - I_{\min}) + I_{\min}$.
- **Rationale:** Simulates subtle non-linear scanner display and transfer function differences. Restricted to $[0.90, 1.10]$ to prevent intensity saturation. `[INFERRED]`

### 3.5 Technique E: Mild Gaussian Noise
- **Parameter Range:** Additive zero-mean Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = 0.05$. `[INFERRED]`
- **Probability:** $p = 0.20$.
- **Rationale:** Simulates thermal and electronic radiofrequency coil noise (MRA) and low-dose detector quantum noise (CTA). Applied to only a small fraction ($20\%$) of presentations. `[INFERRED]`

### 3.6 Technique F: Mild Gaussian Blur
- **Parameter Range:** 3D Gaussian kernel (size $3 \times 3 \times 3$) with $\sigma \sim \mathcal{U}(0.50, 0.75)$ voxels. `[INFERRED]`
- **Probability:** $p = 0.20$.
- **Rationale:** Simulates minor patient micro-motion and slice-thickness partial volume averaging. Conservative blur prevents the obliteration of small vascular lesions (< 3 mm). `[INFERRED]`

---

## 4. Over-Augmentation Safeguards & Anti-Stacking Rules

To prevent compounding artifacts that degrade delicate vascular features:
1. **Single Affine Grid Pass:** 3D rotation and 3D translation are combined into a single homogeneous affine transformation matrix $A \in \mathbb{R}^{4 \times 4}$. The volume is resampled using `torch.nn.functional.grid_sample` exactly once, preventing double trilinear interpolation blurring. `[INFERRED]`
2. **Mutual Exclusion Rule:** Noise and blur are **mutually exclusive** per sample presentation ($P(\text{noise} \cap \text{blur}) = 0$). If both are randomly sampled, only one is applied. `[INFERRED]`
3. **Numeric Boundary Validation:** Every transformed tensor is verified for $\text{isfinite}()$ (asserting 0 NaNs, 0 Infs). `[INFERRED]`

---

## 5. Explicitly Rejected Augmentations & Scientific Justifications

| Excluded Augmentation | Scientific Justification for Exclusion |
| :--- | :--- |
| **1. Scaling (Zoom in/out)** | DATA_SURVEY.md lists 0.8–1.2x as candidate, but full aneurysm-size distribution is marked `[REQUIRES VALIDATION]`. Rescaling could shrink small aneurysms below detection threshold or artificially inflate large ones. |
| **2. Multiplicative Intensity Scaling** | Raw intensity distribution is uncalibrated between MRA and CTA; multiplying raw signal does not reflect documented scanner variance. |
| **3. Low-Resolution Simulation** | Excluded pending initial augmentation findings; introduces multi-axial slice thickness degradation. |
| **4. Left/Right Flipping** | **STRICTLY FORBIDDEN**. TopAneu vessel and aneurysm locations are lateralized (e.g. Left vs. Right ICA/MCA/PCA). Flipping coordinates without a validated label-swap table creates false anatomical associations. |
| **5. Elastic Deformation** | Insufficient evidence that arbitrary B-spline or displacement deformation fields preserve realistic intracranial vascular topology. |
| **6. Cutout / Random Erasing** | Risk of obliterating a small aneurysm while retaining positive presence label, introducing label noise. |
| **7. Mixup / CutMix** | Incompatible with multi-label anatomical segmentation and multi-aneurysm semantics. |
| **8. Synthetic Aneurysm Insertion**| Out of scope; introduces unvalidated generative artifacts. |

---

## 6. Computational & Hardware Plan

- **Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8,150.6 MiB VRAM). `[MEASURED]`
- **Execution Envelope:** Batch size 1, gradient accumulation 8, mixed precision (AMP FP16).
- **Peak VRAM Budget:** Target $< 7,000\text{ MiB}$ (Phase 4 baseline was 6,568 MiB).
- **Expected Training Duration:** $\sim 8\text{ to }10\text{ minutes}$ for 3 epochs.
