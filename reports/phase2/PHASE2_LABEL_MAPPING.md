# Phase 2: Explicit TopAneu ↔ P3 Semantic & Anatomical Mapping Matrix

**Project:** Brain Atlas / Intracranial Aneurysm Detection  
**Evaluation:** TopAneu (52 Locations, 36 Vessels) to P3 (13 Arterial Classes) Alignment  
**Date:** September 12, 2026  

---

## 1. Mapping Methodology & Type Definitions

To prevent artificial inflation of alignment metrics, mappings are strictly categorized into five mutually exclusive types:
- **EXACT:** Direct one-to-one anatomical equivalence (e.g., TopAneu `1.10 BA tip` $\leftrightarrow$ P3 `Basilar Tip`).
- **MERGED:** Multiple fine-grained TopAneu structures correspond to a single broader P3 territory (e.g., TopAneu `R-M1`, `R-M2`, `R-M3` $\to$ P3 `Right MCA`).
- **SPLIT:** A single TopAneu structure spans multiple distinct P3 categories.
- **RELATED:** Anatomically contiguous or branching structures lacking exact boundary correspondence (e.g., TopAneu `R-ICA C7-Pcom-junction` $\leftrightarrow$ P3 `Right PCom` vs `Right Supraclinoid ICA`).
- **NO_DIRECT_EQUIVALENT:** Anatomical variations (e.g., `3rd-A2` median callosal artery) not recognized in P3's standard circle of Willis taxonomy.

---

## 2. Aneurysm Location Mapping Table (52 TopAneu Locations)

| TopAneu ID | TopAneu Location Name | P3 ID | P3 Class Name | Mapping Type | Confidence | Clinical / Anatomical Notes |
| :---: | :--- | :---: | :--- | :---: | :---: | :--- |
| 1 | `R-1.1 VA trunk` | 1 | Other Posterior Circulation | **MERGED** | HIGH | P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation. |
| 2 | `L-1.1 VA trunk` | 1 | Other Posterior Circulation | **MERGED** | HIGH | P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation. |
| 3 | `R-1.2 PICA trunk` | 1 | Other Posterior Circulation | **MERGED** | HIGH | P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation. |
| 4 | `L-1.2 PICA trunk` | 1 | Other Posterior Circulation | **MERGED** | HIGH | P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation. |
| 5 | `R-1.3 VA-PICA junction` | 1 | Other Posterior Circulation | **MERGED** | HIGH | P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation. |
| 6 | `L-1.3 VA-PICA junction` | 1 | Other Posterior Circulation | **MERGED** | HIGH | P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation. |
| 7 | `1.4 BA trunk` | 1 | Other Posterior Circulation | **MERGED** | HIGH | P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation. |
| 8 | `1.5 VA-BA junction` | 1 | Other Posterior Circulation | **MERGED** | HIGH | P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation. |
| 9 | `R-1.6 AICA trunk` | 1 | Other Posterior Circulation | **MERGED** | HIGH | P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation. |
| 10 | `L-1.6 AICA trunk` | 1 | Other Posterior Circulation | **MERGED** | HIGH | P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation. |
| 11 | `R-1.7 BA-AICA junction` | 1 | Other Posterior Circulation | **MERGED** | HIGH | P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation. |
| 12 | `L-1.7 BA-AICA junction` | 1 | Other Posterior Circulation | **MERGED** | HIGH | P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation. |
| 13 | `R-1.8 SCA trunk` | 1 | Other Posterior Circulation | **MERGED** | HIGH | P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation. |
| 14 | `L-1.8 SCA trunk` | 1 | Other Posterior Circulation | **MERGED** | HIGH | P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation. |
| 15 | `R-1.9 BA-SCA junction` | 1 | Other Posterior Circulation | **MERGED** | HIGH | P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation. |
| 16 | `L-1.9 BA-SCA junction` | 1 | Other Posterior Circulation | **MERGED** | HIGH | P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation. |
| 17 | `1.10 BA tip` | 2 | Basilar Tip | **EXACT** | HIGH | Exact anatomical match for basilar bifurcation / apex aneurysm. |
| 18 | `R-2.1 P1P2` | 1 | Other Posterior Circulation | **MERGED** | HIGH | TopAneu distinguishes PCA segments (P1-P4); P3 groups all PCA branches into Other Posterior Circulation. |
| 19 | `L-2.1 P1P2` | 1 | Other Posterior Circulation | **MERGED** | HIGH | TopAneu distinguishes PCA segments (P1-P4); P3 groups all PCA branches into Other Posterior Circulation. |
| 20 | `R-2.2 P3P4` | 1 | Other Posterior Circulation | **MERGED** | HIGH | TopAneu distinguishes PCA segments (P1-P4); P3 groups all PCA branches into Other Posterior Circulation. |
| 21 | `L-2.2 P3P4` | 1 | Other Posterior Circulation | **MERGED** | HIGH | TopAneu distinguishes PCA segments (P1-P4); P3 groups all PCA branches into Other Posterior Circulation. |
| 22 | `R-3.1 ICA infraclinoid C1-C5` | 5 | Right Infraclinoid Internal Carotid Artery | **EXACT** | HIGH | Exact anatomical match for cervical/petrous/cavernous/clinoid segments. |
| 23 | `L-3.1 ICA infraclinoid C1-C5` | 6 | Left Infraclinoid Internal Carotid Artery | **EXACT** | HIGH | Exact anatomical match for cervical/petrous/cavernous/clinoid segments. |
| 24 | `R-3.2 ICA C6-OA-junction` | 7 | Right Supraclinoid Internal Carotid Artery | **MERGED** | HIGH | TopAneu subdivides supraclinoid ICA into C6-OA, C6-nonOA, C7-nonBranch, and C7-terminus; P3 combines these into Supraclinoid ICA. |
| 25 | `L-3.2 ICA C6-OA-junction` | 8 | Left Supraclinoid Internal Carotid Artery | **MERGED** | HIGH | TopAneu subdivides supraclinoid ICA into C6-OA, C6-nonOA, C7-nonBranch, and C7-terminus; P3 combines these into Supraclinoid ICA. |
| 26 | `R-3.3 ICA C6-nonOA` | 7 | Right Supraclinoid Internal Carotid Artery | **MERGED** | HIGH | TopAneu subdivides supraclinoid ICA into C6-OA, C6-nonOA, C7-nonBranch, and C7-terminus; P3 combines these into Supraclinoid ICA. |
| 27 | `L-3.3 ICA C6-nonOA` | 8 | Left Supraclinoid Internal Carotid Artery | **MERGED** | HIGH | TopAneu subdivides supraclinoid ICA into C6-OA, C6-nonOA, C7-nonBranch, and C7-terminus; P3 combines these into Supraclinoid ICA. |
| 28 | `R-3.4 ICA C7-Pcom-junction` | 3 | Right Posterior Communicating Artery | **RELATED** | MEDIUM | TopAneu considers this the ICA junction; P3 annotates PCom aneurysms under Posterior Communicating Artery. Also closely related to Supraclinoid ICA. |
| 29 | `L-3.4 ICA C7-Pcom-junction` | 4 | Left Posterior Communicating Artery | **RELATED** | MEDIUM | TopAneu considers this the ICA junction; P3 annotates PCom aneurysms under Posterior Communicating Artery. Also closely related to Supraclinoid ICA. |
| 30 | `R-3.5 ICA C7-AChA-junction` | 7 | Right Supraclinoid Internal Carotid Artery | **MERGED** | HIGH | Anterior choroidal junction is part of C7 supraclinoid segment in P3 taxonomy. |
| 31 | `L-3.5 ICA C7-AChA-junction` | 8 | Left Supraclinoid Internal Carotid Artery | **MERGED** | HIGH | Anterior choroidal junction is part of C7 supraclinoid segment in P3 taxonomy. |
| 32 | `R-3.6 ICA C7-nonBranch` | 7 | Right Supraclinoid Internal Carotid Artery | **MERGED** | HIGH | TopAneu subdivides supraclinoid ICA into C6-OA, C6-nonOA, C7-nonBranch, and C7-terminus; P3 combines these into Supraclinoid ICA. |
| 33 | `L-3.6 ICA C7-nonBranch` | 8 | Left Supraclinoid Internal Carotid Artery | **MERGED** | HIGH | TopAneu subdivides supraclinoid ICA into C6-OA, C6-nonOA, C7-nonBranch, and C7-terminus; P3 combines these into Supraclinoid ICA. |
| 34 | `R-3.7 ICA C7-terminus` | 7 | Right Supraclinoid Internal Carotid Artery | **MERGED** | HIGH | TopAneu subdivides supraclinoid ICA into C6-OA, C6-nonOA, C7-nonBranch, and C7-terminus; P3 combines these into Supraclinoid ICA. |
| 35 | `L-3.7 ICA C7-terminus` | 8 | Left Supraclinoid Internal Carotid Artery | **MERGED** | HIGH | TopAneu subdivides supraclinoid ICA into C6-OA, C6-nonOA, C7-nonBranch, and C7-terminus; P3 combines these into Supraclinoid ICA. |
| 36 | `4.1 Acom complex` | 13 | Anterior Communicating Artery | **EXACT** | HIGH | Exact anatomical match for ACom complex aneurysms. |
| 37 | `R-4.2 A1` | 11 | Right Anterior Cerebral Artery | **MERGED** | HIGH | TopAneu divides ACA into A1, A2, A3, and distal branches; P3 pools all ipsilateral ACA segments into Right/Left ACA. |
| 38 | `L-4.2 A1` | 12 | Left Anterior Cerebral Artery | **MERGED** | HIGH | TopAneu divides ACA into A1, A2, A3, and distal branches; P3 pools all ipsilateral ACA segments into Right/Left ACA. |
| 39 | `R-4.3 A2` | 11 | Right Anterior Cerebral Artery | **MERGED** | HIGH | TopAneu divides ACA into A1, A2, A3, and distal branches; P3 pools all ipsilateral ACA segments into Right/Left ACA. |
| 40 | `L-4.3 A2` | 12 | Left Anterior Cerebral Artery | **MERGED** | HIGH | TopAneu divides ACA into A1, A2, A3, and distal branches; P3 pools all ipsilateral ACA segments into Right/Left ACA. |
| 41 | `R-4.4 A3` | 11 | Right Anterior Cerebral Artery | **MERGED** | HIGH | TopAneu divides ACA into A1, A2, A3, and distal branches; P3 pools all ipsilateral ACA segments into Right/Left ACA. |
| 42 | `L-4.4 A3` | 12 | Left Anterior Cerebral Artery | **MERGED** | HIGH | TopAneu divides ACA into A1, A2, A3, and distal branches; P3 pools all ipsilateral ACA segments into Right/Left ACA. |
| 43 | `R-4.5 Distal ACA branches` | 11 | Right Anterior Cerebral Artery | **MERGED** | HIGH | TopAneu divides ACA into A1, A2, A3, and distal branches; P3 pools all ipsilateral ACA segments into Right/Left ACA. |
| 44 | `L-4.5 Distal ACA branches` | 12 | Left Anterior Cerebral Artery | **MERGED** | HIGH | TopAneu divides ACA into A1, A2, A3, and distal branches; P3 pools all ipsilateral ACA segments into Right/Left ACA. |
| 45 | `R-5.1 M1 trunk` | 9 | Right Middle Cerebral Artery | **MERGED** | HIGH | TopAneu divides MCA into M1 trunk, bifurcation, junction, and distal branches; P3 pools all into Right/Left MCA. |
| 46 | `L-5.1 M1 trunk` | 10 | Left Middle Cerebral Artery | **MERGED** | HIGH | TopAneu divides MCA into M1 trunk, bifurcation, junction, and distal branches; P3 pools all into Right/Left MCA. |
| 47 | `R-5.2 M1 early bifurcation` | 9 | Right Middle Cerebral Artery | **MERGED** | HIGH | TopAneu divides MCA into M1 trunk, bifurcation, junction, and distal branches; P3 pools all into Right/Left MCA. |
| 48 | `L-5.2 M1 early bifurcation` | 10 | Left Middle Cerebral Artery | **MERGED** | HIGH | TopAneu divides MCA into M1 trunk, bifurcation, junction, and distal branches; P3 pools all into Right/Left MCA. |
| 49 | `R-5.3 M1-M2 junction` | 9 | Right Middle Cerebral Artery | **MERGED** | HIGH | TopAneu divides MCA into M1 trunk, bifurcation, junction, and distal branches; P3 pools all into Right/Left MCA. |
| 50 | `L-5.3 M1-M2 junction` | 10 | Left Middle Cerebral Artery | **MERGED** | HIGH | TopAneu divides MCA into M1 trunk, bifurcation, junction, and distal branches; P3 pools all into Right/Left MCA. |
| 51 | `R-5.3 Distal-M2M3` | 9 | Right Middle Cerebral Artery | **MERGED** | HIGH | TopAneu divides MCA into M1 trunk, bifurcation, junction, and distal branches; P3 pools all into Right/Left MCA. |
| 52 | `L-5.3 Distal-M2M3` | 10 | Left Middle Cerebral Artery | **MERGED** | HIGH | TopAneu divides MCA into M1 trunk, bifurcation, junction, and distal branches; P3 pools all into Right/Left MCA. |

---

## 3. Vascular Tree Mapping Table (36 TopAneu Vessels)

| TopAneu ID | TopAneu Vessel Name | P3 ID | P3 Class Name | Mapping Type | Confidence | Clinical / Anatomical Notes |
| :---: | :--- | :---: | :--- | :---: | :---: | :--- |
| 1 | `BA` | 1 | Other Posterior Circulation | **RELATED** | HIGH | Basilar artery trunk maps to Other Posterior Circulation in P3; apex maps to Basilar Tip. |
| 2 | `R-P1P2` | 1 | Other Posterior Circulation | **MERGED** | HIGH | Vertebral, cerebellar, and posterior cerebral arteries merged into Other Posterior Circulation. |
| 3 | `L-P1P2` | 1 | Other Posterior Circulation | **MERGED** | HIGH | Vertebral, cerebellar, and posterior cerebral arteries merged into Other Posterior Circulation. |
| 4 | `R-ICA-C6-C7` | 7 | Right Supraclinoid Internal Carotid Artery | **EXACT** | HIGH | Exact match for supraclinoid ICA. |
| 5 | `R-M1` | 9 | Right Middle Cerebral Artery | **MERGED** | HIGH | M1, M2, M3 combined into Right MCA. |
| 6 | `L-ICA-C6-C7` | 8 | Left Supraclinoid Internal Carotid Artery | **EXACT** | HIGH | Exact match for supraclinoid ICA. |
| 7 | `L-M1` | 10 | Left Middle Cerebral Artery | **MERGED** | HIGH | M1, M2, M3 combined into Left MCA. |
| 8 | `R-Pcom` | 3 | Right Posterior Communicating Artery | **EXACT** | HIGH | Exact match for right PCom. |
| 9 | `L-Pcom` | 4 | Left Posterior Communicating Artery | **EXACT** | HIGH | Exact match for left PCom. |
| 10 | `Acom` | 13 | Anterior Communicating Artery | **EXACT** | HIGH | Exact match for ACom artery. |
| 11 | `R-A1A2` | 11 | Right Anterior Cerebral Artery | **MERGED** | HIGH | A1, A2, A3 combined into Right ACA. |
| 12 | `L-A1A2` | 12 | Left Anterior Cerebral Artery | **MERGED** | HIGH | A1, A2, A3 combined into Left ACA. |
| 13 | `R-A3` | 11 | Right Anterior Cerebral Artery | **MERGED** | HIGH | A1, A2, A3 combined into Right ACA. |
| 14 | `L-A3` | 12 | Left Anterior Cerebral Artery | **MERGED** | HIGH | A1, A2, A3 combined into Left ACA. |
| 15 | `3rd-A2` | N/A | NO_DIRECT_EQUIVALENT | **NO_DIRECT_EQUIVALENT** | HIGH | Anomalous median anterior cerebral artery variant; not present in P3 standard anatomy. |
| 16 | `3rd-A3` | N/A | NO_DIRECT_EQUIVALENT | **NO_DIRECT_EQUIVALENT** | HIGH | Anomalous median anterior cerebral artery variant; not present in P3 standard anatomy. |
| 17 | `R-M2` | 9 | Right Middle Cerebral Artery | **MERGED** | HIGH | M1, M2, M3 combined into Right MCA. |
| 18 | `R-M3` | 9 | Right Middle Cerebral Artery | **MERGED** | HIGH | M1, M2, M3 combined into Right MCA. |
| 19 | `L-M2` | 10 | Left Middle Cerebral Artery | **MERGED** | HIGH | M1, M2, M3 combined into Left MCA. |
| 20 | `L-M3` | 10 | Left Middle Cerebral Artery | **MERGED** | HIGH | M1, M2, M3 combined into Left MCA. |
| 21 | `R-P3P4` | 1 | Other Posterior Circulation | **MERGED** | HIGH | Vertebral, cerebellar, and posterior cerebral arteries merged into Other Posterior Circulation. |
| 22 | `L-P3P4` | 1 | Other Posterior Circulation | **MERGED** | HIGH | Vertebral, cerebellar, and posterior cerebral arteries merged into Other Posterior Circulation. |
| 23 | `R-VA` | 1 | Other Posterior Circulation | **MERGED** | HIGH | Vertebral, cerebellar, and posterior cerebral arteries merged into Other Posterior Circulation. |
| 24 | `L-VA` | 1 | Other Posterior Circulation | **MERGED** | HIGH | Vertebral, cerebellar, and posterior cerebral arteries merged into Other Posterior Circulation. |
| 25 | `R-SCA` | 1 | Other Posterior Circulation | **MERGED** | HIGH | Vertebral, cerebellar, and posterior cerebral arteries merged into Other Posterior Circulation. |
| 26 | `L-SCA` | 1 | Other Posterior Circulation | **MERGED** | HIGH | Vertebral, cerebellar, and posterior cerebral arteries merged into Other Posterior Circulation. |
| 27 | `R-AICA` | 1 | Other Posterior Circulation | **MERGED** | HIGH | Vertebral, cerebellar, and posterior cerebral arteries merged into Other Posterior Circulation. |
| 28 | `L-AICA` | 1 | Other Posterior Circulation | **MERGED** | HIGH | Vertebral, cerebellar, and posterior cerebral arteries merged into Other Posterior Circulation. |
| 29 | `R-PICA` | 1 | Other Posterior Circulation | **MERGED** | HIGH | Vertebral, cerebellar, and posterior cerebral arteries merged into Other Posterior Circulation. |
| 30 | `L-PICA` | 1 | Other Posterior Circulation | **MERGED** | HIGH | Vertebral, cerebellar, and posterior cerebral arteries merged into Other Posterior Circulation. |
| 31 | `R-AChA` | 7 | Right Supraclinoid Internal Carotid Artery | **RELATED** | MEDIUM | Small anterior choroidal and ophthalmic branches arise from C6/C7 ICA; P3 does not separate micro-branches. |
| 32 | `L-AChA` | 8 | Left Supraclinoid Internal Carotid Artery | **RELATED** | MEDIUM | Small anterior choroidal and ophthalmic branches arise from C6/C7 ICA; P3 does not separate micro-branches. |
| 33 | `R-OA` | 7 | Right Supraclinoid Internal Carotid Artery | **RELATED** | MEDIUM | Small anterior choroidal and ophthalmic branches arise from C6/C7 ICA; P3 does not separate micro-branches. |
| 34 | `L-OA` | 8 | Left Supraclinoid Internal Carotid Artery | **RELATED** | MEDIUM | Small anterior choroidal and ophthalmic branches arise from C6/C7 ICA; P3 does not separate micro-branches. |
| 35 | `R-ICA-C1-C5` | 5 | Right Infraclinoid Internal Carotid Artery | **EXACT** | HIGH | Exact match for infraclinoid ICA. |
| 36 | `L-ICA-C1-C5` | 6 | Left Infraclinoid Internal Carotid Artery | **EXACT** | HIGH | Exact match for infraclinoid ICA. |
