# Phase 4 Completion Report: Independent P3 Recreation, Repository Reorganization, and TopAneu-26 Fine-Tuning

**Project:** Brain Atlas / Intracranial Aneurysm Research  
**Phase:** Phase 4 — Independent Architecture Recreation, Repository Reorganization, and Baseline Fine-Tuning  
**Date:** September 14, 2026  
**Evaluator:** Google DeepMind / Antigravity Agentic Assistant  

---

## 1. Executive Summary

Phase 4 achieves complete technical autonomy from the cloned `RSNA2025_Intracranial-Aneurysm-Detection` repository and external unmaintained packages (`dynamic_network_architectures`). We independently implemented the full 109.36M parameter P3 architecture in modular, native PyTorch under `src/models/p3/`, empirically verified its parameter compatibility against the official competition checkpoint (100% key match, $0$ missing, $0$ unexpected, all specified empirical verification criteria passed), established a leak-free patient-level train/validation/locked-test split for TopAneu-26, eliminated the external repository dependency, built a reproducible configuration-driven training pipeline tailored to local laptop GPU hardware (RTX 5050 8GB VRAM), and conducted comparative benchmark evaluation.

---

## 2. Milestone Achievement & Verification Matrix

| Milestone Task | Specification | Realized Result | Status |
| :--- | :--- | :--- | :---: |
| **1. Repository Audit & Reorganization** | Relocate scattered root files; classify permanent vs temporary | Moved 30 files into `reports/baseline/`, `reports/phase2/`, `reports/phase3/`; clean layout established | **COMPLETE** |
| **2. Independent P3 Recreation** | Native PyTorch implementation in `src/models/p3/` | Modular implementation (`blocks.py`, `head.py`, `pooling.py`, `model.py`) | **COMPLETE** |
| **3. Checkpoint Strict Verification** | Match 109,359,299 params, 583 tensors, strict loading | 109,359,299 params, 583 tensors, 0 missing, 0 unexpected; numerical forward-pass equivalence verified (max divergence $\le 4.14 \times 10^{-7}$) | **COMPLETE** |
| **4. Remove RSNA2025 Dependency** | Purge `RSNA2025/` folder and scratch caches; retain provenance | Deleted `RSNA2025/` and scratch build caches; preserved provenance in `references/p3/`; 100% independent | **COMPLETE** |
| **5. TopAneu-26 Split Protocol** | Patient-level stratified split (408 unique patients) | 70% Train (290 scans), 15% Val (63 scans), 15% Test (62 scans); 0 patient leakage; locked test set | **COMPLETE** |
| **6. Reproducible Training Pipeline** | Config-driven, AMP, gradient accumulation, metric logging | `src/training/trainer.py`, `configs/p4_p3_finetune.yaml`, `scripts/train_p3.py` | **COMPLETE** |
| **7. Local Fine-Tuning Execution** | Train P3 on RTX 5050 Laptop GPU (8GB VRAM) | Experiment `P4-P3-001` executed within 6.57 GB VRAM envelope at 100% GPU utilization | **COMPLETE** |
| **8. Comparative Benchmark Evaluation**| Compare States A, B, and C on locked test split | Primary matched-cohort comparison (State B vs State C on locked $N=62$ test set) | **COMPLETE** |
| **9. Unit Testing & Documentation** | Tests in `tests/`, updated README, completion documentation | Automated test suite (5/5 passing), updated `README.md`, authoritative reports | **COMPLETE** |

---

## 3. Final Repository Architecture

The repository is organized according to standard production research practices:

```text
brain-atlas-model/
├── README.md                           # Master project documentation
├── LICENSE                             # License
├── configs/                            # Declarative training and experiment configs
│   └── p4_p3_finetune.yaml             # Experiment P4-P3-001 configuration
├── src/                                # Reusable modular library
│   ├── data/                           # Data loading and preprocessing (TopAneuDataset)
│   ├── models/                         # Neural network architectures
│   │   └── p3/                         # Independent P3 native implementation (P3Architecture)
│   ├── training/                       # Optimization and training loops (P3Trainer)
│   ├── evaluation/                     # Metric evaluation suite (P3Evaluator)
│   └── xai/                            # Explainability tools and hooks
├── scripts/                            # Standalone executable CLI commands
│   ├── verify_p3_checkpoint.py         # Checkpoint validation CLI
│   ├── create_split.py                 # Patient-level stratified split generator
│   ├── precache_dataset.py             # Multi-core volume pre-cacher
│   ├── train_p3.py                     # Fine-tuning CLI
│   └── evaluate_p3.py                  # Evaluation CLI
├── experiments/                        # Persistent experiment artifacts
│   ├── splits/                         # Machine-readable split definitions
│   │   ├── topaneu_v1.csv              # Patient-level split assignments
│   │   └── topaneu_v1_summary.json     # Split distribution summaries
│   └── results/                        # Saved checkpoints, metrics JSONs, logs
├── reports/                            # Permanent scientific documentation
│   ├── baseline/                       # Phase 1 baseline reports & predictions
│   ├── phase2/                         # Phase 2 anatomical alignment reports & tables
│   ├── phase3/                         # Phase 3 explainability & faithfulness reports
│   └── phase4/                         # Phase 4 audit, split protocol, & validation
├── references/                         # External reference specifications & provenance
│   └── p3/                             # PROVENANCE.md, ARCHITECTURE_SPEC.md, plans.json
├── tests/                              # Automated pytest unit test suite (5/5 passing)
├── topaneu_release/                    # TopAneu-26 multimodal 3D dataset
└── scratch/                            # Ephemeral cache storage (gitignored)
    ├── cache_224/                      # Cached 224^3 preprocessed float32 tensors
    └── checkpoints/                    # Official P3 pretrained weights
```

---

## 4. Hardware Resource Profile & Memory Strategy

All experimentation was performed on consumer-grade laptop hardware:
- **Device:** NVIDIA GeForce RTX 5050 Laptop GPU (8,150.6 MiB VRAM)
- **Host System:** Windows AMD64, 24 CPU cores
- **Software Stack:** Python 3.13.1, PyTorch build `2.14.0+cu130` (compiled with CUDA 13.0, NVIDIA driver 592.19 reporting CUDA 13.1 support)

### Engineering Constraints & Optimizations:
1. **Fine-Tuning Parameter Scope:** P4-P3-001 performed partial fine-tuning of the independently implemented P3 architecture, freezing early spatial encoder stages 0–3 and optimizing deeper semantic stages 4–5 together with the relevant cross-attention/classification components (84,998,211 trainable parameters / 77.72% of the 109,359,299 total parameters; `only_forward_cls=True`). Full multi-task decoder fine-tuning was not performed because of the local VRAM constraint. Peak allocated VRAM was 6,568 MiB (well within the 8,150.6 MiB budget).
2. **Gradient Accumulation:** Physical batch size 1 with gradient accumulation steps = 8 achieves an effective batch size of 8 without memory overhead.
3. **Pre-caching Acceleration:** 100% of the 415 scans were pre-resampled to $(224, 224, 224)$ 1mm isotropic equivalent and cached as float32 `.pt` tensors, accelerating batch loading to $< 20\text{ ms}$ per volume.

---

## 5. Epistemic Classification of Findings

### OBSERVED (Directly Verified via Code Execution & Measurement)
- Our native P3 implementation matches the official architecture across all 109,359,299 parameters and 583 tensor keys.
- Numerical forward-pass equivalence was verified, with a maximum absolute output divergence of $4.14 \times 10^{-7}$ across the tested real clinical cases.
- The codebase executes all training, inference, and testing workflows with zero imports or file dependencies on `RSNA2025/`.
- TopAneu-26 contains 415 scans from 408 unique patients (7 Center-4 patients have longitudinal follow-up scans).
- The patient-level split contains zero patient overlap across Train ($N=290$), Val ($N=63$), and Locked Test ($N=62$) partitions.
- Fine-tuning substantially improved performance on the locked TopAneu-26 test partition relative to the independently implemented P3 initialized with the official weights (ROC-AUC: 0.6046 $\rightarrow$ 0.8928, PR-AUC: 0.8369 $\rightarrow$ 0.9626).
- At the fixed $\tau=0.50$ threshold, fine-tuning substantially increased sensitivity (68.89% $\rightarrow$ 95.56%) while specificity decreased slightly (35.29% $\rightarrow$ 29.41%) on the locked test partition. Observed false negatives directly decreased from 14 to 2.
- The Brier score decreased from 0.2714 to 0.1294 on the locked test partition, indicating lower probabilistic prediction error under this metric.
- State C produced 12 false-positive predictions on the locked test partition at $\tau=0.50$.
- All 5 automated unit tests pass in 3.24 seconds.

### INFERRED (Scientifically Grounded Conclusions)
- The difference between State A (whole dataset, $N=415$) and State B (locked test partition, $N=62$) is consistent with differences in cohort composition, sampling variation, and class prevalence between the two evaluation populations. The available numerical equivalence checks provide no observed evidence of an implementation discrepancy, although exhaustive case-level equivalence between the historical State A run and the native implementation was not performed.
- Pre-caching $224^3$ volumes eliminated CPU bottlenecks during training, enabling continuous GPU compute utilization.
- The improvement is consistent with adaptation to distributional differences between the P3 training domain and TopAneu-26.
- Some false positives may reflect vascular structures whose imaging characteristics resemble aneurysms.

### UNKNOWN (Open Questions & Unverified Hypotheses)
- The present experiment does not isolate scanner, intensity, demographic, acquisition, or other individual sources of domain shift.
- Specific morphological causes of the remaining false positives have not been established by the Phase 4 experiment.
- Multi-task decoder fine-tuning (vascular segmentation + territory segmentation) was not attempted due to local VRAM limits (>50 GB requirement); its impact on TopAneu-26 remains untested locally.
- Longitudinal scan time intervals for Center-4 patients are not annotated in dataset metadata.
- Generalization to external multi-center cohorts beyond TopAneu-26 remains UNKNOWN until multi-institutional validation is performed.
- Formal calibration assessment was not performed in Phase 4 and remains outside the demonstrated scope.

---

## 6. Fine-Tuning Performance & Comparative Benchmark Summary

### 6.1 State Comparison on Locked Test Partition ($N=62$)

> **Cohort Clarification:**  
> State A is retained as the historical full-cohort P3 reference. Because State A was evaluated on $N=415$ cases while States B and C were evaluated on the locked $N=62$ test partition, State A is not used for the primary matched-cohort comparison. The primary assessment of fine-tuning is the State B versus State C comparison on the identical locked test set.

| Metric | State A (Historical Full-Cohort Reference, Unmatched) | State B (Independent P3 + Official Weights) | State C (Fine-Tuned `P4-P3-001`) | Primary Delta (State C vs State B) |
| :--- | :---: | :---: | :---: | :---: |
| **Cohort Definition** | Full Dataset ($N=415$) | Locked Test Partition ($N=62$) | Locked Test Partition ($N=62$) | Identical Locked Cohort |
| **Accuracy** | 62.41% | 59.68% | **77.42%** | **+17.74%** |
| **Sensitivity (Recall)** | 68.42% | 68.89% | **95.56%** | **+26.67%** |
| **Specificity** | 45.95% | 35.29% | 29.41% | -5.88% |
| **Precision (PPV)** | 77.61% | 73.81% | **78.18%** | **+4.37%** |
| **F1-Score** | 0.7273 | 0.7126 | **0.8600** | **+0.1474** |
| **ROC-AUC** | 0.6388 | 0.6046 | **0.8928** | **+0.2882** |
| **PR-AUC** | 0.8529 | 0.8369 | **0.9626** | **+0.1257** |
| **Brier Score** | 0.2491 | 0.2714 | **0.1294** | **-0.1420** |
| **Confusion Matrix** | $\text{TP}=208, \text{FP}=60, \text{TN}=51, \text{FN}=96$ | $\text{TP}=31, \text{FP}=11, \text{TN}=6, \text{FN}=14$ | **$\text{TP}=43, \text{FP}=12, \text{TN}=5, \text{FN}=2$** | FN: $14 \rightarrow 2$ |

---

## 7. Exact Reproduction Commands

All Phase 4 workflows are fully deterministic and reproducible via the following CLI commands:

```bash
# 1. Run Automated Unit Test Suite
pytest tests/ -v

# 2. Verify P3 Architecture Strict Checkpoint Loading & Equivalence
python scripts/verify_p3_checkpoint.py

# 3. Generate Patient-Level Stratified Dataset Splits
python scripts/create_split.py

# 4. Pre-cache TopAneu-26 Scans to 224^3 Tensors
python scripts/precache_dataset.py

# 5. Evaluate Baseline Model (State B: Official Checkpoint on Independent Code)
python scripts/evaluate_p3.py --checkpoint scratch/checkpoints/Dataset660_26classes_resize224_4661/onlyMirror01_lr4e3_100epochs_ps224/fold_0/checkpoint_final.pth --split test --state_name State_B_Official

# 6. Run Fine-Tuning Pipeline (Experiment P4-P3-001)
python scripts/train_p3.py --config configs/p4_p3_finetune.yaml

# 7. Evaluate Fine-Tuned Model (State C: P4-P3-001 on Locked Test Set)
python scripts/evaluate_p3.py --checkpoint experiments/results/P4-P3-001/best_checkpoint.pth --split test --state_name State_C_Finetuned
```

---

## 8. Phase 4 Stop Condition Verification

All specified Phase 4 conditions are satisfied:
1. Repository organized $\rightarrow$ **PASS**
2. Independent P3 implemented $\rightarrow$ **PASS**
3. Official P3 checkpoint verified $\rightarrow$ **PASS**
4. `RSNA2025/` runtime dependency removed $\rightarrow$ **PASS**
5. TopAneu train/val/test protocol established $\rightarrow$ **PASS**
6. P3 training pipeline created $\rightarrow$ **PASS**
7. P3 fine-tuned on TopAneu-26 $\rightarrow$ **PASS**
8. Evaluation completed across States A, B, and C $\rightarrow$ **PASS**
9. Comprehensive documentation and tests completed $\rightarrow$ **PASS**

**STOP CONDITION MET.** We halt before dataset augmentation, architecture modifications, anatomical loss constraints, and NLP.
