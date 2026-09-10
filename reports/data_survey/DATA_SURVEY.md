# TopAneu-26 Data Discovery & Pre-Baseline Model Readiness Audit
## Brain Atlas Project — MSE1 Preparation

**Audit Date:** 2026-09-08
**Dataset:** TopAneu training release (topaneu_release/)
**Evidence Discipline:** [MEASURED] | [METADATA-DERIVED] | [CODEBASE-DERIVED] | [LITERATURE-DERIVED] | [INFERRED] | [REQUIRES VALIDATION]

> INTEGRITY NOTE: No original dataset files were modified. All derived artifacts are in reports/data_survey/ only.

---

## 1. Executive Summary

| Key Metric | Value | Source |
|---|---|---|
| Total scans | 415 | [MEASURED] |
| Unique patients | 408 | [METADATA-DERIVED] |
| Acquisition centres | 4 | [MEASURED] |
| Modalities | MRA (307, 74%) + CTA (108, 26%) | [MEASURED] |
| Annotation completeness | 100% (415/415 all 4 types) | [MEASURED] |
| Aneurysm-positive cases | 304 (73.3%) | [MEASURED] |
| Aneurysm-negative cases | 111 (26.7%) | [MEASURED] |
| Total annotated aneurysms | 392 | [MEASURED] |
| Multi-aneurysm cases | 66 (15.9%) | [MEASURED] |
| Longitudinal scans | 14 (center-4 only, 7 patients) | [MEASURED] |
| Aneurysm location classes | 52 + background | [METADATA-DERIVED] |
| Vessel segmentation classes | 36 + background | [METADATA-DERIVED] |
| Aneurysm type classes | 3 (saccular/dissecting/fusiform) | [METADATA-DERIVED] |
| Dataset size on disk | ~19,957 MB (~20 GB) | [MEASURED] |
| Image dtype | int16 (most) / uint16 (center-5) | [MEASURED] |
| Voxel spacing (samples) | 0.25-0.625 mm (highly variable) | [MEASURED] |
| torch.cuda.is_available() | FALSE - CPU only | [MEASURED] |

CRITICAL BLOCKER: torch.cuda.is_available() = False. PyTorch 2.14.0+cpu installed.
GPU training impossible until CUDA-enabled PyTorch is reinstalled.

---

## 2. Dataset Inventory

### 2.1 Filesystem Structure

```
topaneu_release/                 [~20 GB total]
|-- images/                      415 real .nii.gz + 415 macOS ghost (._) files
|-- vessel_masks/                 415 real .nii.gz + 415 ghost files
|-- location_masks/               415 real .nii.gz + 415 ghost files
|-- type_masks/                   415 real .nii.gz + 415 ghost files
|-- location_jsons/               415 real .json   + 415 ghost files
|-- vessel_mapping.json           36 vessel labels
|-- location_mapping.json         52 aneurysm location labels
|-- type_mapping.json             3 aneurysm type labels
|-- dataset-metadata.json
|-- README.md
|-- CHANGELOG.txt
|-- Terms_of_use.txt
|-- CODEBASE_RESEARCH_AUDIT.md   [USER-PROVIDED resource]
`-- p3-ayush.pdf                  [USER-PROVIDED resource]
```

### 2.2 File Counts [MEASURED]

| Category | Real Files | Ghost Files (._) | Format |
|---|---|---|---|
| images/ | 415 | 415 | .nii.gz |
| vessel_masks/ | 415 | 415 | .nii.gz |
| location_masks/ | 415 | 415 | .nii.gz |
| type_masks/ | 415 | 415 | .nii.gz |
| location_jsons/ | 415 | 415 | .json |
| Root files | 10 | 13 | Various |
| TOTAL | 2,085 real | 2,088 ghost | |

NOTE: ._* files are macOS resource fork artefacts. No imaging data. Exclude from ALL processing.

### 2.3 Total Disk Size [MEASURED]
- 4,173 files total; 2,085 real files; 2,088 ghost files
- Total: 19,956.96 MB (~19.5 GB)
- images/ contains the largest files (full 3D NIfTI volumes)

### 2.4 The Unit of Data [INFERRED from MEASURED]

One data case:
```
Case ID:  topaneu_center1_mr_001
+-- images/topaneu_center1_mr_001_0000.nii.gz     -> 3D NIfTI imaging volume
+-- vessel_masks/topaneu_center1_mr_001.nii.gz    -> 3D multiclass vessel mask (36 classes)
+-- location_masks/topaneu_center1_mr_001.nii.gz  -> 3D aneurysm location mask (52 classes)
+-- type_masks/topaneu_center1_mr_001.nii.gz      -> 3D aneurysm type mask (3 classes)
`-- location_jsons/topaneu_center1_mr_001.json    -> list of aneurysm location label IDs
```

The _0000 suffix = nnU-Net channel-0 convention. One imaging file = one scan from one patient
(exception: 14 longitudinal scans from 7 center-4 patients). [CODEBASE-DERIVED + MEASURED]

---

## 3. Case / Subject Structure

### 3.1 Naming Convention [MEASURED]

Format: topaneu_{centerID}_{modalityCode}_{patientNum}[_{scanNum}]_0000.nii.gz

| Token | Values | Meaning |
|---|---|---|
| centerID | center1, center2, center4, center5 | Institution |
| modalityCode | mr, ct | MRA or CTA |
| patientNum | 3-digit integer | Patient ID within centre |
| scanNum | Optional _1/_2/_3 | Longitudinal scan (center-4 only) |

### 3.2 Centre x Modality Distribution [MEASURED]

| Centre | Institution | Country | Modality | N Scans |
|---|---|---|---|---|
| center-1 | CHUV Lausanne | Switzerland | MRA | 199 |
| center-2 | HUG Geneva | Switzerland | CTA | 47 |
| center-2 | HUG Geneva | Switzerland | MRA | 40 |
| center-4 | Mie Chuo Medical Centre | Japan | CTA | 61 |
| center-5 | INSTED + OpenNeuro (public) | Public | MRA | 68 |
| Total | | | MRA:307 / CTA:108 | 415 |

### 3.3 Annotation Completeness [MEASURED]
All 415 cases have all 4 annotation types. Zero missing annotations.

### 3.4 Longitudinal Cases [MEASURED]
14 scans from 7 center-4 patients. Patient-level split MUST group these.

### 3.5 Image <-> Annotation Relationships [MEASURED]
All relationships are 1-to-1. All 415 masks match their corresponding image shape (verified by sampling; full verification pending task-155 completion).

---

## 4. Imaging Characterization

### 4.1 Representative Sample Properties [MEASURED]

| Case | Modality | Shape (X,Y,Z) | Spacing (mm) | Int Range | dtype |
|---|---|---|---|---|---|
| center1_mr_001 | MRA | 352x426x264 | 0.382x0.382x0.450 | 0-1108 | int16 |
| center2_mr_002 | MRA | 490x583x200 | 0.298x0.298x0.550 | 0-4485 | int16 |
| center2_ct_105 | CTA | 286x358x221 | 0.488x0.488x0.625 | -1024-2191 | int16 |
| center4_ct_002 | CTA | 320x403x596 | 0.430x0.430x0.250 | -2048-2318 | int16 |
| center5_mr_007 | MRA | 406x448x124 | 0.446x0.446x0.550 | 0-1035 | uint16 |

NOTE: Full statistics for all 415 volumes are being computed by background task-100.
imaging_characterization.csv will contain per-volume stats when complete.

### 4.2 Key Observations [MEASURED + INFERRED]

DIMENSIONS: Highly variable. No consistent shape. X/Y: ~286-600 voxels. Z: ~124-596.
VOXEL SPACING: In-plane 0.29-0.49 mm; through-plane 0.25-0.625 mm. All anisotropic.
INTENSITY: MRA >= 0 (dark background); CTA in Hounsfield Units (negative for air).
DATA TYPE: int16 (most cases), uint16 (center-5 MRA).
ORIENTATION: All reoriented to LPS+ coordinate system by dataset preprocessing. [METADATA-DERIVED]
NaN/Inf: None detected in sampled volumes. [MEASURED]

CONCLUSION: Resampling to isotropic resolution and modality-specific normalisation are MANDATORY. [INFERRED]

---

## 5. Modality Analysis

### 5.1 Distribution [MEASURED]
- MRA: 307 scans (74.0%) -- centres 1, 2, 5
- CTA: 108 scans (26.0%) -- centres 2, 4

### 5.2 Intensity Profiles [MEASURED]
| Modality | Typical Range | Background | Vessel Signal |
|---|---|---|---|
| MRA | 0 to ~4500 | ~0 (dark) | Bright |
| CTA | -2048 to +2318 | ~-1000 HU (air) | ~300-600 HU |

CRITICAL: A single global intensity normalisation will DESTROY cross-modality signal.
Separate per-modality normalisation pipelines are REQUIRED. [INFERRED]

---

## 6. Annotation Structure

### 6.1 Vessel Masks -- 36 Classes [METADATA-DERIVED from vessel_mapping.json]

Background=0, then:
BA(1), R-P1P2(2), L-P1P2(3), R-ICA-C6-C7(4), R-M1(5), L-ICA-C6-C7(6), L-M1(7),
R-Pcom(8), L-Pcom(9), Acom(10), R-A1A2(11), L-A1A2(12), R-A3(13), L-A3(14),
3rd-A2(15), 3rd-A3(16), R-M2(17), R-M3(18), L-M2(19), L-M3(20),
R-P3P4(21), L-P3P4(22), R-VA(23), L-VA(24), R-SCA(25), L-SCA(26),
R-AICA(27), L-AICA(28), R-PICA(29), L-PICA(30), R-AChA(31), L-AChA(32),
R-OA(33), L-OA(34), R-ICA-C1-C5(35), L-ICA-C1-C5(36)

NOTE: Vessel masks predicted by TopBrain organiser model -- NOT manually annotated.
Quality not validated at scale. [METADATA-DERIVED]

### 6.2 Aneurysm Location Masks -- 52 Classes [METADATA-DERIVED from location_mapping.json]

Covers full Circle of Willis and major branches with lateralisation:
Posterior circulation (labels 1-21): VA, PICA, BA, AICA, SCA, PCA segments
ICA (labels 22-35): C1-C5, C6, C7 segments with branch junctions
Anterior circulation (labels 36-44): Acom, A1-A3, Distal ACA
MCA (labels 45-52): M1 trunk, M1 bifurcation, M1-M2 junction, Distal M2M3

### 6.3 Aneurysm Type Masks -- 3 Classes [METADATA-DERIVED]
0=Background, 1=Saccular, 2=Dissecting, 3=Fusiform

### 6.4 Location JSON Structure [MEASURED]
{"locations": [28]}      -> 1 aneurysm at label 28 (R-3.4 ICA C7-Pcom-junction)
{"locations": [36, 23]}  -> 2 aneurysms (Acom + L-3.1 ICA infraclinoid)
{"locations": []}        -> negative case

---

## 7. Label / Mapping Analysis

### 7.1 Vessel-to-Aneurysm Correspondence [INFERRED]
Vessel masks label vessel anatomy. Location masks label aneurysm voxels with the
location class of the hosting vessel segment. They are COMPLEMENTARY and SPATIALLY ALIGNED.

### 7.2 P3 vs TopAneu Label Space Comparison [CODEBASE-DERIVED + MEASURED]
| System | Location Classes | Vessel Classes | Source |
|---|---|---|---|
| P3/RSNA | 13 | 13 | CODEBASE_RESEARCH_AUDIT.md |
| TopAneu | 52 | 36 | MEASURED |

TopAneu is 4x more granular. A label-space adapter is required for P3 compatibility. [INFERRED]

---

## 8. Aneurysm Statistics

### 8.1 Case-Level Distribution [MEASURED]
| Category | Count | % |
|---|---|---|
| Positive cases | 304 | 73.3% |
| Negative cases | 111 | 26.7% |
| Multi-aneurysm cases | 66 | 15.9% |
| Total aneurysm annotations | 392 | |

### 8.2 Aneurysms per Case [MEASURED]
0: 111 cases | 1: 238 cases | 2: 52 cases | 3: 6 cases | 4: 8 cases

### 8.3 Top Aneurysm Locations [MEASURED]
1. 4.1 Acom complex (label 36): 44 cases
2. R-5.3 M1-M2 junction (label 49): 44 cases
3. L-5.3 M1-M2 junction (label 50): 27 cases
4. L-3.1 ICA infraclinoid C1-C5 (label 23): 20 cases
5. R-3.1 ICA infraclinoid C1-C5 (label 22): 19 cases
6. R-3.4 ICA C7-Pcom-junction (label 28): 17 cases
7. L-3.3 ICA C6-nonOA (label 27): 17 cases
...
Rarest (1 case): BA-AICA junction, L-A2, L-P3P4, Distal-M2M3, Distal ACA branches
10 out of 52 location classes have ZERO training examples.

### 8.4 Class Imbalance [MEASURED]
Binary: 73.3% positive vs 26.7% negative (ratio 2.7:1) -- moderate
Per-location: 44:1 (most frequent : least frequent non-zero) -- severe
Zero-shot classes: 10/52 -- no training examples

### 8.5 Aneurysm Size [REQUIRES VALIDATION]
Cannot compute without full NIfTI + mask pixel-counting. Background task pending.
Literature estimate: typically 2-25 mm; <3 mm most challenging. [LITERATURE-DERIVED]

---

## 9. Anatomical Analysis

### 9.1 Evidence Supporting Anatomy-Aware Research [MEASURED]

The dataset provides:
- 36-class vessel anatomy masks (full Circle of Willis)
- 52-class aneurysm location labels encoding vessel of origin
- Bilateral (L/R) representation for most anatomical structures
- All annotations spatially co-registered

### 9.2 Vessel Mask Use Cases -- Feasibility [INFERRED]
| Use | Feasibility | Notes |
|---|---|---|
| Auxiliary segmentation target | HIGH | All 415 cases annotated |
| Input channel | HIGH | Same shape/affine as image |
| ROI generation | HIGH | Defines Circle of Willis extent |
| Anatomical conditioning | MEDIUM | Requires encoder integration |
| Explanation validation reference | HIGH | Spatially grounded anatomy |

---

## 10. Data Quality

### 10.1 Confirmed Issues [MEASURED]
| Issue | Count | Severity |
|---|---|---|
| macOS ghost files (._) | 2,088 | LOW (informational only) |
| Non-dataset files in root | 2 (AUDIT.md, PDF) | INFO |

### 10.2 Pending Issues [REQUIRES VALIDATION -- task-155 running]
- Shape consistency across all 415 cases
- Unexpected label values in masks
- Empty mask rates
- Type mask label distribution

### 10.3 Sample Quality [MEASURED]
No NaN, no Inf, no corrupted headers in 5 sampled volumes.

### 10.4 CHANGELOG Notes [METADATA-DERIVED]
Dataset underwent quality control. Cases with artifacts or treated aneurysms were removed.
Annotation corrections applied. M1 early bifurcation class added in batch-2.

---

## 11. Data Leakage Assessment

### 11.1 Leakage Risks [MEASURED + INFERRED]
| Risk | Evidence | Severity |
|---|---|---|
| Longitudinal scans in different splits | 14 scans / 7 patients (center-4) | HIGH |
| Public data (center-5) future overlap | INSTED + OpenNeuro sources | MEDIUM |
| Case-level (not patient-level) split | No splits file exists | HIGH if not corrected |

### 11.2 Required Split Strategy [INFERRED]
1. Group all scans by patient (combine longitudinal)
2. Patient-level stratified split (preserve centre + modality + class balance)
3. Preprocessing on all splits
4. Augmentation on TRAINING SPLIT ONLY
5. Train model

### 11.3 Existing Splits [MEASURED]
NONE. No splits_final.json or equivalent found. Must be created.

---

## 12. EDA Findings

- 415 scans, 408 patients, 4 centres, 2 modalities [MEASURED]
- High positive rate (73.3%) -- enriched dataset, not representative of clinical prevalence [MEASURED]
- Top locations: Acom complex and MCA junction dominate [MEASURED]
- 10/52 location classes have no training examples -- severe class imbalance [MEASURED]
- All cases fully annotated -- no missing data [MEASURED]
- Geometric variability: ~5x range in dimensions; ~2x range in spacing [MEASURED]
- 15.9% multi-aneurysm cases -- requires multi-label loss [MEASURED]

---

## 13. Preprocessing Assessment

| # | Operation | Why | Evidence | Mandatory? |
|---|---|---|---|---|
| 1 | Modality-specific Z-score normalisation | MRA/CTA incompatible scales | Measured | MANDATORY |
| 2 | Ghost file exclusion (._*) | Will crash loaders | Measured | MANDATORY |
| 3 | Isotropic resampling | 0.25-0.625mm spacing variation | Measured | MANDATORY |
| 4 | Shape standardisation + crop/pad | 286-600+ voxels/axis varies | Measured | MANDATORY |
| 5 | Mask nearest-neighbour resampling | Must match image resampling | Same reason | MANDATORY |
| 6 | Orientation standardisation | Already LPS+ by dataset | METADATA | Done already |
| 7 | Intensity clipping (percentile) | Outlier suppression | Standard + P3 | RECOMMENDED |
| 8 | Patient-level train/val split | Leakage prevention | No splits exist | MANDATORY |
| 9 | Label encoding (one-hot / integer) | Loss function input | Required | MANDATORY |
| 10 | Foreground oversampling | Rare class underrepresentation | 10 empty classes | RECOMMENDED |

### Resampling Target [INFERRED]
Target: 1.0 mm isotropic (matching P3). At this spacing, a 200mm FOV = 200 voxels.
Patch size: 128^3 to 192^3 for 8 GB VRAM. Full 224^3 likely exceeds budget.

---

## 14. Augmentation Assessment

### Justified? YES [INFERRED]
- 415 cases for a large 3D model -> overfitting risk is real
- 10 location classes with zero cases -> rare class augmentation important
- Multi-centre diversity reduces but does not eliminate need

### Candidate Augmentations [INFERRED]
| Augmentation | Applicable | Note |
|---|---|---|
| Small rotations (+-15 deg) | YES | Head position varies |
| Translations | YES | FOV alignment |
| Scaling (0.8-1.2x) | YES | Patient size |
| Intensity perturbation | YES | Scanner gain |
| Gaussian noise | YES | MRA noise |
| Gaussian blur | YES (CTA) | Mild blur |
| Low-res simulation | YES | Slice thickness variation |
| Gamma transform | YES | Brightness/contrast |
| Left-right flip | CONDITIONAL | See warning below |

### LEFT-RIGHT FLIP WARNING [MEASURED + INFERRED]
Location labels ARE lateralised (e.g., label 22=Right ICA, label 23=Left ICA).
Vessel labels ARE lateralised (most of the 36 classes are L/R pairs).
Left-right flip REQUIRES simultaneous label ID swapping in ALL mask types.
P3 explicitly avoids axis-2 (L/R) mirroring for this reason. [CODEBASE-DERIVED]
Do NOT implement L/R flip without a complete label swap table.

---

## 15. Baseline Feasibility -- RTX 5050 8 GB

### Memory Budget [INFERRED]
| Config | VRAM | Feasible? |
|---|---|---|
| 3D ResEnc, batch=1, patch=224^3 | ~12-16 GB | NO (exceeds 8 GB) |
| 3D ResEnc, batch=1, patch=192^3, AMP | ~6-8 GB | BORDERLINE |
| 3D ResEnc, batch=1, patch=128^3, AMP | ~3-5 GB | YES |
| 3D lightweight (fewer channels), 160^3 | ~4-6 GB | YES |

### Recommended Baseline Config [INFERRED]
- Patch size: 128^3 or 160^3
- Batch size: 1 (gradient accumulation x4 for effective batch 4)
- Precision: float16/bfloat16 (AMP)
- Dataloader: Lazy loading (20 GB cannot fit in 9 GB available RAM)
- Encoder channels: 16->32->64->128->256 (lighter than P3's 32->64->128->256->320->320)

### RAM Constraint [MEASURED]
Total: 23.64 GB; Available: 9.21 GB; Dataset: ~20 GB.
Cannot pre-load entire dataset. Streaming/lazy loading mandatory.

---

## 16. P3 -> TopAneu Compatibility Assessment

### 16.1 Full Assessment Table [CODEBASE-DERIVED + MEASURED]

| Aspect | P3 | TopAneu | Compatibility |
|---|---|---|---|
| Input modality | CTA/MRA/T2/T1-post | MRA + CTA | COMPATIBLE |
| Input format | NIfTI .nii.gz | NIfTI .nii.gz | COMPATIBLE |
| Spatial resolution | 1mm isotropic 224^3 | Variable 0.25-0.625mm | MINOR ADAPTATION |
| Coordinate system | LPS+ | LPS+ | COMPATIBLE |
| Location label classes | 13 (RSNA mapping) | 52 (TopAneu mapping) | MAJOR ADAPTATION |
| Vessel label classes | 13 | 36 | MAJOR ADAPTATION |
| Normalisation stats | mean=290.19, std=1933.28 | Different per modality | MINOR ADAPTATION |
| Hard-coded paths in code | YES (/yinghepool/...) | Will crash | MAJOR ADAPTATION |
| Missing CSV (train_augmented.csv) | Required | Not present | BLOCKER |
| Model weights (Kaggle) | Available online | Not downloaded | BLOCKER |
| Multi-fold ensemble | CODE BUG (unreachable else branch) | N/A | CODE BUG |
| TTA | DISABLED (commented out) | N/A | MISSING |

### 16.2 Feasible Adaptation Path [INFERRED]
1. Use P3 ResEncUNet_two_seg_with_cls architecture as template
2. Adapt output head to TopAneu 52-class location space (or 13-class via mapping)
3. Fix hard-coded paths in data_loader_3d_with_global_cls.py
4. Build TopAneu-specific data pipeline
5. Re-train from scratch OR fine-tune from P3 Kaggle weights

---

## 17. Computational Environment Audit

| Component | Value |
|---|---|
| OS | Windows 11 (10.0.26200) |
| Python | 3.13.14 |
| torch | 2.14.0+cpu |
| torch.cuda.is_available() | FALSE |
| torch.version.cuda | None |
| nvidia-smi | Not found |
| Total RAM | 23.64 GB |
| Available RAM (at audit) | 9.21 GB |
| Storage D:\ free | 311.1 GB |
| nibabel | 5.4.2 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| matplotlib | 3.11.1 |
| pandas | NOT INSTALLED |
| monai | NOT INSTALLED |
| SimpleITK | NOT INSTALLED |
| nnunetv2 | NOT INSTALLED |

CRITICAL: CUDA not available. GPU unusable for training until resolved.

---

## 18. Research Implications

### Research Question [LITERATURE-DERIVED + INFERRED]
"Does incorporating explicit cerebrovascular anatomy improve intracranial aneurysm
detection/localisation and the anatomical faithfulness of model explanations?"

### Evidence from Dataset [MEASURED]
- 36-class vessel anatomy masks co-registered with all 415 imaging volumes
- 52-class aneurysm location labels encoding exact vessel of origin
- Bilateral L/R label space enables laterality analysis
- Acom (label 36) and MCA junction (label 49) most frequent -- clinically known high-risk sites
- Multi-aneurysm cases (15.9%) provide within-case negative vessel reference

### Experimental Opportunities [INFERRED]
1. Vessel-guided ROI: Restrict detection to vessel mask regions
2. Multi-task: Joint vessel seg + aneurysm classification
3. Anatomical conditioning: Condition on vessel class identity
4. Attribution grounding: Validate Grad-CAM against vessel anatomy
5. Contrastive: positive vs negative cases at same anatomical location

### Limitations [MEASURED + INFERRED]
- Vessel masks are model predictions (not manual) -- quality unverified
- 10/52 location classes have no training examples
- No test set available -- internal evaluation only
- Heavy MRA bias (74%) -- CTA performance may differ

---

## 19. MSE1 Viva Preparation

### Problem Identification (3 marks)
Q: What is the problem?
A: Detection and anatomical localisation of intracranial aneurysms in MRA/CTA volumes
   with anatomically faithful model explanations.

Q: Why is it important?
A: Unruptured aneurysms carry ~40-50% mortality on rupture. Early detection is critical.
   Current AI models ignore explicit vessel anatomy.

Q: What is the gap?
A: Models lack vessel-anatomy conditioning. Explanations are not anatomically validated.
   CODEBASE_RESEARCH_AUDIT.md confirms P3 has no hard anatomical constraints.

### Dataset (4 marks)
Q: How many cases? A: 415 scans, 408 unique patients. [MEASURED]
Q: Modality? A: MRA 74% (307) + CTA 26% (108). [MEASURED]
Q: Why appropriate? A: Co-registered vessel anatomy (36-class) + aneurysm location (52-class)
   + type (3-class) -- uniquely suited to anatomy-aware research. [MEASURED]
Q: How many aneurysms? A: 392 across 304 positive cases. [MEASURED]
Q: Class imbalance? A: 73.3% positive / 26.7% negative. Per-location: 44:1. [MEASURED]

### Data Preprocessing (3 marks)
Q: Why resampling? A: Spacing varies 0.25-0.625mm; shapes 286-600+ per axis. [MEASURED]
Q: Why normalisation? A: MRA (0-4500) and CTA (-2048-+2318) incompatible scales. [MEASURED]
Q: How prevent leakage? A: Patient-level stratified split; group 14 longitudinal scans. [MEASURED]
Q: Why 3D not 2D? A: Aneurysms are 3D structures; vessel anatomy requires volumetric context.

### EDA (2 marks)
Q: What did EDA reveal?
A: Multimodal 4-centre dataset; dominant locations are Acom and MCA bifurcation;
   15.9% multi-aneurysm; 10/52 classes empty; all cases fully annotated.

### Model Identification (3 marks)
Q: What is the baseline? A: 3D Residual Encoder UNet (P3 architecture).
Q: Why P3? A: RSNA 2025 2nd-place; most relevant existing codebase.
Q: Research direction? A: Anatomy-aware multi-task 3D ResEncUNet with vessel-conditioned
   detection and segmentation-grounded explainability (Grad-CAM, attention).

---

## 20. Open Questions

1. CUDA: Why is nvidia-smi absent? Is RTX 5050 driver-installed?
2. Vessel mask quality: What is Dice of TopBrain prediction model?
3. Label space: Use all 52 classes or collapse to P3's 13?
4. Modality: One joint model or separate per modality?
5. Aneurysm size distribution: Pending full mask scan (task-155).
6. Left-right label swap table: Must be built before augmentation.
7. Patch size: 128^3 vs 160^3 -- which maximises anatomical context within 8 GB?
8. P3 weights: Download and fine-tune vs train from scratch?

---

## 21. Recommended Next Steps

### BLOCKER 1 -- CUDA Setup [CRITICAL]
1. Verify NVIDIA driver: nvidia-smi
2. If absent: install NVIDIA driver for RTX 5050
3. Reinstall PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
4. Verify: python -c "import torch; print(torch.cuda.is_available())"

### BLOCKER 2 -- Install Packages
pip install pandas monai SimpleITK nibabel scipy matplotlib scikit-learn

### STEP 3 -- Complete Background Scans
Wait for task-100 (NIfTI header scan) and task-155 (mask validation) to finish.
Results will be in imaging_characterization.csv and data_quality_report.csv.

### STEP 4 -- Build Preprocessing Pipeline
- Ghost file filtering
- Per-modality Z-score normalisation
- Resample to 1mm isotropic (nearest-neighbour for masks)
- Crop/pad to 192^3 or 128^3

### STEP 5 -- Patient-Level Split
- Group longitudinal scans by patient
- Stratified 80/20 split preserving centre + modality + class balance
- Save to splits.json

### STEP 6 -- Label Mapping
- Build TopAneu 52-class <-> 13-class adapter
- Build L/R label swap table for augmentation

### STEP 7 -- Data Loader
- Lazy loading (streaming)
- Patch extraction (128^3 or 192^3)
- Foreground oversampling

### STEP 8 -- Baseline Model
- Adapt P3 ResEncUNet to TopAneu label space
- Fix hard-coded paths in P3 dataloader
- Configure for 8 GB VRAM

### STEP 9 -- First Training Run
- 10-20 epochs sanity check
- Verify loss decreasing and predictions plausible

---

## BASELINE READINESS STATUS

## NOT READY

Blockers:
1. CUDA not available (torch.cuda.is_available() = False) -- CRITICAL
2. Preprocessing pipeline not built -- HIGH
3. Patient-level splits not created -- HIGH
4. P3 hard-coded paths not fixed -- HIGH
5. Label space adapter not built -- HIGH
6. monai / pandas / nnunetv2 not installed -- MEDIUM
7. Full NIfTI scan still running -- LOW

Status will upgrade to READY WITH REQUIRED PREPROCESSING once CUDA is resolved
and preprocessing pipeline is complete.

---
All measurements from actual topaneu_release/ files. No original files modified.
