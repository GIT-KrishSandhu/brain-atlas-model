# Phase 4: TopAneu-26 Patient-Level Data Split Protocol

**Project:** Brain Atlas / Intracranial Aneurysm Research  
**Investigation Scope:** Specification and Verification of Patient-Level Train/Validation/Test Partitions  
**Dataset:** TopAneu-26 (415 3D volumes across 408 patients)  
**Date:** September 14, 2026  
**Evaluator:** Google DeepMind / Antigravity Agentic Assistant  

---

## 1. Protocol Rationale & Design Objectives

In medical imaging benchmarks, arbitrary scan-level random splitting frequently introduces subtle data leakage when multiple scans from the same subject (longitudinal follow-ups or multimodal CTA/MRA acquisitions) are allocated to both training and evaluation subsets. 

To guarantee scientific rigor:
1. **Zero Patient-Level Leakage:** All scans from any individual patient reside exclusively in exactly one partition (Train, Validation, or Locked Test).
2. **Stratified Allocation:** Stratified allocation was used to maintain approximately similar aneurysm-presence and modality distributions across partitions.
3. **Locked Test Partition:** The test partition ($N=62$ scans) is locked and forbidden from use during intermediate training iterations, hyperparameter sweeps, or model selection.
4. **Reproducibility:** The splitting algorithm is fully deterministic under fixed seed (`seed=42`) and machine-readable at `experiments/splits/topaneu_v1.csv`.

---

## 2. Subject Grouping & Schema Mapping

The TopAneu-26 dataset contains 415 scans across 408 unique patients. Filename schemas map to patient entities via:
$$\text{Schema: } \texttt{topaneu\_\{centerID\}\_\{modality\}\_\{patientID\}}$$

- **Center-4 Longitudinal Scans:** Seven subjects in center-4 have multiple follow-up scans annotated with trailing suffixes (e.g., `topaneu_center4_ct_008_1` and `topaneu_center4_ct_008_2`). The trailing enumeration was stripped to identify the single underlying patient (`center4_008`).
- **Center-2 Multimodal Acquisitions:** Single subjects with both CTA and MRA acquisitions (e.g. `topaneu_center2_ct_002` and `topaneu_center2_mr_002`) share the unified patient identifier (`center2_002`).

Across all 415 files, this reduces to exactly **408 unique patients** (297 positive patients, 111 negative patients).

---

## 3. Partition Distribution & Balance Matrix

Target allocations: **70% Train / 15% Validation / 15% Locked Test** (`seed=42`).

| Metric | Train Split | Validation Split | Locked Test Split | Full Dataset |
| :--- | :---: | :---: | :---: | :---: |
| **Unique Patients** | 284 (69.6%) | 62 (15.2%) | 62 (15.2%) | 408 (100.0%) |
| **Patient Overlap** | 0 | 0 | 0 | **ZERO LEAKAGE** |
| **Total Scans** | 290 (69.9%) | 63 (15.2%) | 62 (14.9%) | 415 (100.0%) |
| **Positive Scans** | 213 | 46 | 45 | 304 |
| **Negative Scans** | 77 | 17 | 17 | 111 |
| **Aneurysm Prevalence** | **73.45%** | **73.02%** | **72.58%** | **73.25%** |
| **MRA Scans** | 213 (73.4%) | 47 (74.6%) | 47 (75.8%) | 307 (74.0%) |
| **CTA Scans** | 77 (26.6%) | 16 (25.4%) | 15 (24.2%) | 108 (26.0%) |

---

## 4. Machine-Readable Artifacts

The split definitions and detailed summaries are preserved under:
- **Split Table:** `experiments/splits/topaneu_v1.csv`  
  *Columns:* `case_id`, `patient_id`, `center_id`, `modality`, `presence`, `num_aneurysms`, `split`
- **Summary Metadata:** `experiments/splits/topaneu_v1_summary.json`

---

## 5. Epistemic Classification

- **OBSERVED:** Exact count of 415 scans and 408 unique patients derived directly from the filesystem and dataset metadata.
- **OBSERVED:** Zero subject overlap confirmed by set-intersection assertions across train ($N=290$), val ($N=63$), and locked test ($N=62$) subsets.
- **INFERRED:** Stratified allocation maintained approximately similar aneurysm-presence and modality distributions across partitions (prevalence between 72.58% and 73.45%), mitigating potential sampling imbalances during evaluation.
- **UNKNOWN:** Clinical follow-up time intervals for center-4 longitudinal scans (not annotated in file metadata).
