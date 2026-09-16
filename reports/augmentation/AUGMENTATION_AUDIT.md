# Data Augmentation Phase: Pipeline Audit & Pre-Experiment Readiness Assessment

**Project:** Brain Atlas / Intracranial Aneurysm Research  
**Phase:** Data Augmentation Phase (Experimental Control on TopAneu-26)  
**Date:** September 16, 2026  
**Evaluator:** Google DeepMind / Antigravity Agentic Assistant  
**Evidence Discipline:** `[MEASURED]`, `[METADATA-DERIVED]`, `[CODEBASE-DERIVED]`, `[LITERATURE-DERIVED]`, `[INFERRED]`, `[REQUIRES VALIDATION]`

---

## 1. Executive Summary & Objective

This audit establishes the pre-experimental state of the TopAneu-26 training pipeline following the completion of Phase 4. Phase 4 established an independent native PyTorch implementation of the 109.36M parameter P3 neural architecture, verified strict numerical equivalence with official weights, constructed a leak-free patient-level data split, established an unaugmented fine-tuned baseline (State C / Experiment `P4-P3-001`), and established a locked evaluation protocol on the TopAneu-26 test cohort ($N=62$).

The objective of this phase is strictly defined:
> **Research Question:** "Does controlled augmentation of the training distribution improve the performance and robustness of the fine-tuned P3 model on TopAneu-26?"

To isolate the impact of training distribution perturbations, all other experimental components are strictly frozen:
- No changes to the model architecture (P3 remains 109,359,299 parameters). `[CODEBASE-DERIVED]`
- No anatomy-conditioned classifier.
- No new attention mechanisms.
- No new loss functions.
- No alteration of the patient-level train/validation/test split (`experiments/splits/topaneu_v1.csv`). `[CODEBASE-DERIVED]`
- No augmentation applied to validation or locked test partitions.

---

## 2. Phase 4 Baseline Pipeline Audit

### 2.1 Codebase Architecture & Modularity
The codebase operates natively in PyTorch with zero dependencies on external unmaintained competition repositories:
- `src/models/p3/`: Modular architecture comprising `blocks.py` (ResidualBlock, ConvDropoutNormNonlin), `head.py` (ClsHead), `pooling.py` (CrossAttentionPooling), and `model.py` (`P3Architecture`). `[CODEBASE-DERIVED]`
- `src/data/dataset.py`: `TopAneuDataset` implementing volume loading, 3D trilinear resampling, standard P3 Z-score normalization, and tensor caching. `[CODEBASE-DERIVED]`
- `src/training/trainer.py`: `P3Trainer` handling optimization via AdamW, CosineAnnealingLR, mixed precision (`torch.amp.autocast(dtype=torch.float16)`), and gradient accumulation. `[CODEBASE-DERIVED]`
- `src/evaluation/evaluator.py`: `P3Evaluator` standardized evaluation suite computing confusion matrix, accuracy, sensitivity, specificity, precision, NPV, F1, ROC-AUC, PR-AUC, and Brier score. `[CODEBASE-DERIVED]`
- `scratch/cache_224/`: Disk cache containing 415 preprocessed `.pt` tensors representing $(1, 224, 224, 224)$ volumes. `[MEASURED]`

### 2.2 Baseline Fine-Tuning Performance (Condition A0: `P4-P3-001`)
The frozen control condition for this phase is State C from Phase 4 (`P4-P3-001`), fine-tuned with 0% data augmentation:
- **Validation Split ($N=63$):** ROC-AUC: 0.9079, PR-AUC: 0.9655, Accuracy: 88.89%, Sensitivity: 93.48%, Specificity: 76.47%, F1: 0.9247, Brier: 0.1104. `[MEASURED]`
- **Locked Test Split ($N=62$):** ROC-AUC: 0.8928, PR-AUC: 0.9626, Accuracy: 77.42%, Sensitivity: 95.56%, Specificity: 29.41%, F1: 0.8600, Brier: 0.1294. `[MEASURED]`
- **Locked Test Confusion Matrix ($\tau = 0.50$):** TP = 43, FP = 12, TN = 5, FN = 2. `[MEASURED]`

---

## 3. Preprocessing, Normalization & Caching Audit

### 3.1 Preprocessing Pipeline
The Phase 4 preprocessing steps for each raw NIfTI volume are:
1. **Raw Volume Loading:** Loaded via `nibabel.load(path).get_fdata().astype(np.float32)`. Raw orientations are already standardized to LPS+ coordinate frame. `[METADATA-DERIVED]`
2. **3D Trilinear Resampling:** Resampled to target dimension $(224, 224, 224)$ equivalent to 1.0 mm isotropic spacing using `scipy.ndimage.zoom(..., order=1)`. `[CODEBASE-DERIVED]`
3. **Z-Score Normalization:** Standardized per volume:
   $$\tilde{I}(x, y, z) = \frac{I(x, y, z) - \mu}{\max(\sigma, 10^{-8})}$$
   where $\mu$ and $\sigma$ are the empirical mean and standard deviation of the resampled volume. `[CODEBASE-DERIVED]`
4. **Cache Serialization:** Cached on disk as `scratch/cache_224/{case_id}.pt` with shape $(1, 224, 224, 224)$. `[MEASURED]`

### 3.2 Modality-Specific Intensity Characteristics
As measured in `DATA_SURVEY.md`:
- **MRA Scans (307 volumes, 74.0%):** Raw intensities range from $0$ to $\sim 4500$, with dark background ($\sim 0$) and bright vascular signal. `[MEASURED]`
- **CTA Scans (108 volumes, 26.0%):** Calibrated in Hounsfield Units (HU) ranging from $-2048$ to $+2318$, with background air ($\sim -1000\text{ HU}$) and vascular contrast ($\sim 300\text{ to }600\text{ HU}$). `[MEASURED]`
- **Normalization Preservation:** Because per-scan Z-score normalization maps both modalities into zero-mean, unit-variance distributions while preserving internal structural contrast, this normalization step remains intact and must not be altered or replaced. `[INFERRED]`

---

## 4. Augmentation Insertion Point Analysis

### 4.1 Insertion Point Location
In Phase 4, `TopAneuDataset.__getitem__` directly returned the cached normalized tensor $X \in \mathbb{R}^{1 \times 224 \times 224 \times 224}$.

The exact insertion point for data augmentation is:
$$\text{Raw Cache Tensor } X \longrightarrow \mathbf{[AUGMENTATION\ PIPELINE]} \longrightarrow X_{\text{aug}} \longrightarrow \text{DataLoader Batch}$$
Specifically:
- Augmentation is applied **on-the-fly** inside `TopAneuDataset.__getitem__` if and only if `split == 'train'` and `transform is not None`. `[CODEBASE-DERIVED]`
- Validation and test loaders are instantiated with `transform=None` (guaranteeing 0% augmentation). `[CODEBASE-DERIVED]`
- Augmentations execute in PyTorch tensor space, leveraging vectorization.

### 4.2 Mathematical Considerations of Insertion on Normalized Volumes
Applying augmentations to Z-score normalized volumes ($\mu \approx 0, \sigma \approx 1$) requires strict mathematical safeguards:
1. **Spatial Transformations (Rotation & Translation):** Coordinate mapping $(x, y, z) \mapsto (x', y', z')$ via `torch.nn.functional.grid_sample`.
   - *Padding Mode:* Using `padding_mode='border'` ensures boundary voxels (which represent background air/tissue at volume margins) are clamped rather than introducing artificial high-contrast zero-step discontinuities. `[INFERRED]`
   - *Single Affine Grid:* Composing rotation and translation into a single 3D affine matrix prevents cumulative interpolation blur caused by multiple sequential resampling steps. `[INFERRED]`
2. **Contrast Variation:** $I_{\text{aug}} = I \times c$ with $c \in [0.90, 1.10]$. Because $I$ has zero mean, this directly modulates the distribution variance without shifting the mean. `[INFERRED]`
3. **Gamma Transformation:** Standard gamma transformation $I^\gamma$ requires non-negative values in $[0, 1]$. To apply gamma safely on normalized real-valued volumes:
   - Rescale dynamically: $I_{\text{scaled}} = \frac{I - I_{\min}}{I_{\max} - I_{\min} + \epsilon} \in [0, 1]$.
   - Apply power law: $I_{\gamma} = (I_{\text{scaled}})^\gamma$.
   - Invert scaling: $I_{\text{aug}} = I_{\gamma} \times (I_{\max} - I_{\min}) + I_{\min}$.
   - This preserves bounds and guarantees zero `NaN` generation. `[INFERRED]`
4. **Gaussian Noise:** Additive perturbation $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = 0.05$ (5% of normalized unit variance). Applied probabilistically ($p = 0.20$). `[INFERRED]`
5. **Gaussian Blur:** 3D spatial Gaussian smoothing with $\sigma \in [0.5, 0.75]$ voxels. Applied probabilistically ($p = 0.20$). `[INFERRED]`

---

## 5. Dataset Constraints & Lateralization Risks

### 5.1 TopAneu-26 Dataset Inventory
- Total Scans: 415 `[MEASURED]`
- Unique Patients: 408 `[METADATA-DERIVED]`
- Centers: 4 acquisition centers (Center-1: MRA; Center-2: MRA+CTA; Center-4: CTA longitudinal; Center-5: MRA) `[MEASURED]`
- Split: 290 train (284 patients), 63 val (62 patients), 62 test (62 patients) `[MEASURED]`
- Zero Patient Leakage: Fully verified by set intersection between splits. `[MEASURED]`

### 5.2 Lateralized Anatomy & Mirroring Prohibition
- TopAneu vessel segmentation labels (36 classes) and aneurysm location labels (52 classes) are explicitly lateralized (e.g. Left ICA vs Right ICA, Left MCA vs Right MCA). `[METADATA-DERIVED]`
- Axis-2 (Left/Right) flipping is **strictly prohibited** because it would invert anatomical laterality unless an exhaustive, verified label-swapping lookup table is applied to all masks and location labels. `[CODEBASE-DERIVED + INFERRED]`

### 5.3 Multi-Aneurysm Cases
- The dataset contains **66 multi-aneurysm cases** (up to 4 distinct aneurysms per scan). `[MEASURED]`
- Augmentations must not assume single-aneurysm semantics. Spatial transformations must uniformly map all aneurysms within a case without altering relative geometric alignments. `[INFERRED]`

---

## 6. Audit Conclusion & Readiness Status

1. **Pipeline Autonomy:** Independent P3 codebase is fully self-contained and free of external competition dependencies.
2. **Pre-cached Data:** 415 preprocessed volumes at $(224, 224, 224)$ are cached in `scratch/cache_224/`.
3. **Hardware Readiness:** GPU execution confirmed on RTX 5050 Laptop GPU (8GB VRAM) with CUDA 13.0 and AMP FP16.
4. **Safety Verification:** All proposed augmentations are bounded, conservative, and mathematically structured for normalized volumes.
5. **Status:** APPROVED FOR EXPERIMENTAL PLAN EXECUTION.
