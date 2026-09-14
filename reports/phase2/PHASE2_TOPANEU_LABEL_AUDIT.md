# Phase 2: TopAneu Ground Truth Label & Annotation Geometry Audit

**Project:** Brain Atlas / Intracranial Aneurysm Detection  
**Document:** TopAneu Ground Truth Label Audit  
**Date:** September 12, 2026  
**Evaluator:** Google DeepMind / Antigravity Agentic Assistant  

---

## 1. Directory Structure & File Taxonomy

The TopAneu dataset (`topaneu_release`) provides rich multi-level ground-truth annotations across 415 3D neurovascular scans (304 positive, 111 negative):

```
topaneu_release/
├── images/           # Raw 3D scans (*_0000.nii.gz)
├── vessel_masks/     # 3D multiclass intracranial vessel segmentations (*.nii.gz)
├── location_masks/   # 3D multiclass aneurysm location segmentations (*.nii.gz)
├── type_masks/       # 3D multiclass aneurysm morphological type segmentations (*.nii.gz)
├── location_jsons/   # Metadata listing aneurysm location IDs per scan (*.json)
├── vessel_mapping.json    # Dictionary of 36 vessel integer labels
├── location_mapping.json  # Dictionary of 52 aneurysm location integer labels
└── type_mapping.json      # Dictionary of 3 morphology integer labels
```

---

## 2. Integer Label Enumeration

### 2.1 Vessel Labels (`vessel_mapping.json`) — 36 Structures
| ID | Code Name | Anatomical Definition |
| :---: | :--- | :--- |
| **0** | `background` | Background non-vascular parenchyma / air |
| **1** | `BA` | Basilar Artery |
| **2** | `R-P1P2` | Right Posterior Cerebral Artery (P1/P2 segments) |
| **3** | `L-P1P2` | Left Posterior Cerebral Artery (P1/P2 segments) |
| **4** | `R-ICA-C6-C7` | Right Internal Carotid Artery (Ophthalmic & Communicating segments) |
| **5** | `R-M1` | Right Middle Cerebral Artery (M1 horizontal segment) |
| **6** | `L-ICA-C6-C7` | Left Internal Carotid Artery (Ophthalmic & Communicating segments) |
| **7** | `L-M1` | Left Middle Cerebral Artery (M1 horizontal segment) |
| **8** | `R-Pcom` | Right Posterior Communicating Artery |
| **9** | `L-Pcom` | Left Posterior Communicating Artery |
| **10** | `Acom` | Anterior Communicating Artery |
| **11** | `R-A1A2` | Right Anterior Cerebral Artery (A1/A2 segments) |
| **12** | `L-A1A2` | Left Anterior Cerebral Artery (A1/A2 segments) |
| **13** | `R-A3` | Right Anterior Cerebral Artery (A3 pericallosal segment) |
| **14** | `L-A3` | Left Anterior Cerebral Artery (A3 pericallosal segment) |
| **15** | `3rd-A2` | Median Anterior Cerebral Artery (Accessory A2 anatomical variant) |
| **16** | `3rd-A3` | Median Anterior Cerebral Artery (Accessory A3 anatomical variant) |
| **17** | `R-M2` | Right Middle Cerebral Artery (M2 insular segment) |
| **18** | `R-M3` | Right Middle Cerebral Artery (M3 opercular segment) |
| **19** | `L-M2` | Left Middle Cerebral Artery (M2 insular segment) |
| **20** | `L-M3` | Left Middle Cerebral Artery (M3 opercular segment) |
| **21** | `R-P3P4` | Right Posterior Cerebral Artery (P3/P4 cortical segments) |
| **22** | `L-P3P4` | Left Posterior Cerebral Artery (P3/P4 cortical segments) |
| **23** | `R-VA` | Right Vertebral Artery |
| **24** | `L-VA` | Left Vertebral Artery |
| **25** | `R-SCA` | Right Superior Cerebellar Artery |
| **26** | `L-SCA` | Left Superior Cerebellar Artery |
| **27** | `R-AICA` | Right Anterior Inferior Cerebellar Artery |
| **28** | `L-AICA` | Left Anterior Inferior Cerebellar Artery |
| **29** | `R-PICA` | Right Posterior Inferior Cerebellar Artery |
| **30** | `L-PICA` | Left Posterior Inferior Cerebellar Artery |
| **31** | `R-AChA` | Right Anterior Choroidal Artery |
| **32** | `L-AChA` | Left Anterior Choroidal Artery |
| **33** | `R-OA` | Right Ophthalmic Artery |
| **34** | `L-OA` | Left Ophthalmic Artery |
| **35** | `R-ICA-C1-C5` | Right Internal Carotid Artery (Cervical, Petrous, Cavernous, Clinoid) |
| **36** | `L-ICA-C1-C5` | Left Internal Carotid Artery (Cervical, Petrous, Cavernous, Clinoid) |

---

### 2.2 Aneurysm Location Labels (`location_mapping.json`) — 52 Locations
TopAneu segments aneurysm lesions by exact anatomical site, encoded with integer IDs 1 to 52:
- **Vertebrobasilar & Cerebellar (IDs 1–16):** Right/Left VA trunk (1, 2), PICA trunk (3, 4), VA-PICA junction (5, 6), BA trunk (7), VA-BA junction (8), AICA trunk (9, 10), BA-AICA junction (11, 12), SCA trunk (13, 14), BA-SCA junction (15, 16).
- **Basilar Tip (ID 17):** `1.10 BA tip`.
- **Posterior Cerebral (IDs 18–21):** Right/Left P1P2 (18, 19), Right/Left P3P4 (20, 21).
- **Infraclinoid ICA (IDs 22–23):** Right/Left ICA infraclinoid C1–C5 (22, 23).
- **Supraclinoid ICA (IDs 24–35):** Right/Left ICA C6-OA junction (24, 25), C6-nonOA (26, 27), C7-Pcom junction (28, 29), C7-AChA junction (30, 31), C7-nonBranch (32, 33), C7-terminus (34, 35).
- **Anterior Communicating (ID 36):** `4.1 Acom complex`.
- **Anterior Cerebral (IDs 37–44):** Right/Left A1 (37, 38), A2 (39, 40), A3 (41, 42), Distal ACA branches (43, 44).
- **Middle Cerebral (IDs 45–52):** Right/Left M1 trunk (45, 46), M1 early bifurcation (47, 48), M1-M2 junction (49, 50), Distal-M2M3 (51, 52).

---

### 2.3 Morphology Type Labels (`type_mapping.json`) — 3 Classes
| ID | Type Name | Description |
| :---: | :--- | :--- |
| **0** | `background` | Non-aneurysm tissue |
| **1** | `saccular` | Saccular / Berry aneurysm |
| **2** | `dissecting` | Dissecting aneurysm |
| **3** | `fusiform` | Fusiform / Dolichoectatic aneurysm |

---

## 3. Spatial Geometry & Overlap Analysis

### 3.1 Spatial Consistency Across Modalities
Inspection across NIfTI headers confirmed:
- For every case, `image`, `vessel_mask`, `location_mask`, and `type_mask` have **strictly identical dimensions, identical voxel spacings, and 100% matching affine transformation matrices** (`np.allclose(img_affine, mask_affine) == True`).
- No internal orientation discrepancies exist within the TopAneu file collections.

### 3.2 Label Overlap Characteristics
1. **`location_masks` vs. `type_masks`:**
   - Possess **identical non-zero spatial support** ($\{v \mid \text{loc}(v) > 0\} \equiv \{v \mid \text{type}(v) > 0\}$).
   - `location_masks` assign the anatomical position integer ID ($1..52$).
   - `type_masks` assign the morphology integer ID ($1..3$).
2. **`vessel_masks` vs. `location_masks`:**
   - **Partially Overlapping:** The aneurysm neck and originating arterial lumen are frequently co-segmented in both `vessel_masks` and `location_masks` (e.g., in case 017, 456 of 576 aneurysm voxels overlap with the host vessel).
   - In contrast, P3's Decoder 1 was trained under a **mutually exclusive label model** where the aneurysm dome replaces the vessel wall at that location.
