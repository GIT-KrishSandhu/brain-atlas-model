# Phase 4: Repository Audit & Restructuring Specification

**Project:** Brain Atlas / Intracranial Aneurysm Research  
**Investigation Scope:** Pre-restructuring Audit of Codebase, Reports, Checkpoints, and Dependencies  
**Date:** September 14, 2026  
**Evaluator:** Google DeepMind / Antigravity Agentic Assistant  

---

## 1. Executive Summary & Inventory

Prior to making modifications, a comprehensive audit of all files in the repository was performed. The repository currently contains historical baseline documentation, Phase 2 anatomical alignment evaluations, Phase 3 explainability (XAI) audits, ad-hoc Python scripts in `scratch/`, and an external cloned directory (`RSNA2025_Intracranial-Aneurysm-Detection/`).

The purpose of this audit is to classify every artifact, define its permanent target location, and eliminate runtime dependencies on external code.

---

## 2. Artifact Classification Taxonomy

| Category | Definition | Repository Files Identified | Planned Action |
| :--- | :--- | :--- | :--- |
| **Permanent Research Code** | Modular libraries, model definitions, preprocessing, training, inference | None in root (previously scattered in `scratch/` and imported from `RSNA2025/`) | Implement natively in `src/` (`src/data/`, `src/models/p3/`, `src/training/`, `src/inference/`, `src/evaluation/`, `src/xai/`) |
| **Executable Scripts** | CLI entry points for verification, training, evaluation, split generation | Ad-hoc scripts in `scratch/*.py` | Codify into standard CLI entrypoints in `scripts/` |
| **Configurations** | Hyperparameters, split definitions, training configs | Embedded as dictionaries in Python scripts | Centralize into declarative YAML files in `configs/` |
| **Historical Baseline Reports** | Phase 1 environment, forward pass, B0 reference, whole-dataset evaluation | `BASELINE_*.md`, `GPU_BASELINE_TEST.md`, `P3_PRETRAINED_BASELINE_REPORT.md`, `CHECKPOINT_COMPATIBILITY.md`, `DATA_AVAILABLE.md`, `P3_END_TO_END_VALIDATION.md`, `PRETRAINED_CHECKPOINT_STATUS.md`, `STAGE1_BASELINE_STATUS.md` | Move to `reports/baseline/` |
| **Phase 2 Reports & Tables** | Dual-decoder segmentation, location alignment, label mappings | `PHASE2_ANATOMICAL_ALIGNMENT_REPORT.md`, `PHASE2_LABEL_MAPPING.md`, `PHASE2_LABEL_MAPPING.csv`, `PHASE2_P3_SEGMENTATION_MAPPING.md`, `PHASE2_SPATIAL_ALIGNMENT.md`, `PHASE2_TOPANEU_LABEL_AUDIT.md`, `phase2_location_alignment.csv`, `phase2_segmentation_metrics.csv` | Move to `reports/phase2/` |
| **Phase 3 Reports & Tables** | Explainability, attribution benchmark, faithfulness perturbation | `PHASE3_XAI_REPORT.md`, `PHASE3_FAITHFULNESS_REPORT.md`, `PHASE3_XAI_FAILURE_ANALYSIS.md`, `PHASE3_XAI_ARCHITECTURE_AUDIT.md`, `PHASE3_XAI_ENVIRONMENT.md`, `PHASE3_XAI_METHOD_COMPARISON.csv`, `PHASE3_XAI_METRICS.csv`, `PHASE3_FAITHFULNESS_RESULTS.csv` | Move to `reports/phase3/` |
| **Data Survey Reports** | Dataset discovery, imaging summaries, case relationships | `reports/data_survey/*` | Retain under `reports/data_survey/` |
| **External Reference Material** | Original P3 plans, paper provenance, citations | Cloned repo `RSNA2025_Intracranial-Aneurysm-Detection/` | Extract provenance and plans to `references/p3/`, then delete `RSNA2025/` after verification |
| **Temporary Scratch & Caches** | Diagnostic logs, pilot runs, tarballs, pip cache | `scratch/dna_src/`, `scratch/pip_cache/`, `scratch/phase3_pilot_outputs/`, `scratch/dynamic_network_architectures-0.3.1.tar.gz` | Clean and delete after native reimplementation |
| **Checkpoints** | Official pretrained P3 weights | `scratch/checkpoints/Dataset660_26classes_resize224_4661/.../checkpoint_final.pth` | Retain in `scratch/checkpoints/` (gitignored) |

---

## 3. Detailed File Migration Plan

### 3.1 Migration of Historical Reports

To eliminate clutter in the repository root while preserving all historical scientific findings:

1. **Move to `reports/baseline/`:**
   - `BASELINE_ENVIRONMENT.md`
   - `BASELINE_EXPERIMENT_REPORT.md`
   - `BASELINE_FORWARD_PASS.md`
   - `BASELINE_RESULTS.md`
   - `BASELINE_RUN_LOG.md`
   - `CHECKPOINT_COMPATIBILITY.md`
   - `DATA_AVAILABLE.md`
   - `GPU_BASELINE_TEST.md`
   - `P3_END_TO_END_VALIDATION.md`
   - `P3_PRETRAINED_BASELINE_REPORT.md`
   - `PRETRAINED_CHECKPOINT_STATUS.md`
   - `STAGE1_BASELINE_STATUS.md`
   - `p3_pretrained_topaneu_predictions.csv`
   - `stage1_topaneu_predictions.csv`

2. **Move to `reports/phase2/`:**
   - `PHASE2_ANATOMICAL_ALIGNMENT_REPORT.md`
   - `PHASE2_LABEL_MAPPING.csv`
   - `PHASE2_LABEL_MAPPING.md`
   - `PHASE2_P3_SEGMENTATION_MAPPING.md`
   - `PHASE2_SPATIAL_ALIGNMENT.md`
   - `PHASE2_TOPANEU_LABEL_AUDIT.md`
   - `phase2_location_alignment.csv`
   - `phase2_segmentation_metrics.csv`

3. **Move to `reports/phase3/`:**
   - `PHASE3_FAITHFULNESS_REPORT.md`
   - `PHASE3_FAITHFULNESS_RESULTS.csv`
   - `PHASE3_XAI_ARCHITECTURE_AUDIT.md`
   - `PHASE3_XAI_ENVIRONMENT.md`
   - `PHASE3_XAI_FAILURE_ANALYSIS.md`
   - `PHASE3_XAI_METHOD_COMPARISON.csv`
   - `PHASE3_XAI_METRICS.csv`
   - `PHASE3_XAI_REPORT.md`

### 3.2 Reference & Provenance Archive (`references/p3/`)
- `references/p3/PROVENANCE.md`: Document original solution authors (Pengcheng Shi et al.), RSNA 2025 competition ranking (2nd place), arXiv paper reference (arXiv:2606.26706), and checkpoint SHA256 checksum.
- `references/p3/plans.json`: Extracted official network plans configuration (`Dataset660_26classes_resize224_4661`).
- `references/p3/ARCHITECTURE_SPEC.md`: Detailed specification of the 6-stage encoder, bottleneck dimensions, cross-attention pooling, and dual-decoder heads.

---

## 4. Execution Guardrails
- No report content or historical empirical numbers will be altered during file relocation.
- Relocation will be performed via Git or file system moves.
- `RSNA2025_Intracranial-Aneurysm-Detection/` will not be removed until Step 4 (checkpoint verification of our independent implementation) has passed strict verification.

---

## 5. Post-Reorganization Status & Final Structure

All proposed migration actions have been successfully executed:
- **Baseline Reports:** 14 files migrated into `reports/baseline/`.
- **Phase 2 Artifacts:** 8 files migrated into `reports/phase2/`.
- **Phase 3 Artifacts:** 8 files migrated into `reports/phase3/`.
- **External Dependencies Purged:** `RSNA2025_Intracranial-Aneurysm-Detection/` and scratch build caches (`scratch/dna_src/`, `scratch/pip_cache/`) were completely deleted after independent verification passed.
- **Reference & Provenance Retained:** `references/p3/PROVENANCE.md`, `references/p3/ARCHITECTURE_SPEC.md`, and `references/p3/plans.json` are intact.
- **Independent Codebase:** Pure native PyTorch architecture in `src/models/p3/` verified against official checkpoint with zero missing/unexpected keys.
- **Operational Verification:** Full test suite in `tests/` passes with 100% test pass rate.

