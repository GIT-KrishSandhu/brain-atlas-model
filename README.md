# Brain Atlas Model: 3D Intracranial Aneurysm Detection & Segmentation

An independent, reproducible research codebase for multimodal 3D intracranial aneurysm classification, localization, and segmentation using the P3 architecture on the clinical **TopAneu-26** dataset.

---

## 1. Project Overview & Research Objectives

Intracranial aneurysms are life-threatening vascular abnormalities requiring rapid, precise 3D detection and territorial localization across diverse imaging modalities (Time-of-Flight MRA and Contrast-Enhanced CTA).

This repository independently recreates, benchmarks, and fine-tunes the **P3 architecture** (the 2nd-place solution of the RSNA 2025 Intracranial Aneurysm Detection Challenge by Pengcheng Shi et al., arXiv:2606.26706) without external dependencies on competition notebooks or unmaintained packages.

### Completed Research Phases
- **Phase 1 (Baseline Verification):** Verified GPU execution environment (CUDA 13.0, PyTorch 2.14.0 on RTX 5050 8GB VRAM) and sanity baseline (B0).
- **Phase 2 (Anatomical Alignment):** Evaluated dual-decoder spatial alignment across 13 arterial territories and 15 vascular classes.
- **Phase 3 (Explainability Audit):** Audited model attention, integrated gradients, and perturbation faithfulness across cross-attention pooling heads.
- **Phase 4 (Architecture Recreation & Fine-Tuning):** Recreated the P3 architecture in native modular PyTorch, verified strict compatibility with official pretrained weights (109.36M params, 583 tensors), purged external runtime dependencies, established a leak-free patient-level split protocol, and built a local fine-tuning pipeline.

---

## 2. Distinction Between Model States

To maintain rigorous scientific clarity, this project distinguishes between three distinct model states:

1. **State A (Historical Full-Cohort Reference, Unmatched):**  
   Pretrained official P3 Stage-2 checkpoint evaluated on the full TopAneu-26 dataset ($N=415$) without local fine-tuning.
   - Whole-dataset ($N=415$): Accuracy 62.41%, Sensitivity 68.42%, Specificity 45.95%, ROC-AUC 0.6388, PR-AUC 0.8529.
   - *Note:* State A is retained as the historical full-cohort P3 reference. Because State A was evaluated on $N=415$ cases while States B and C were evaluated on the locked $N=62$ test partition, State A is not used for the primary matched-cohort comparison.
2. **State B (Independent P3 Architecture with Official Weights):**  
   Our native modular PyTorch implementation (`src/models/p3/`) loaded strictly with official weights. Numerical forward-pass equivalence was verified, with a maximum absolute output divergence of $4.14 \times 10^{-7}$ across the tested real clinical cases.
   - Locked Test Split ($N=62$): Accuracy 59.68%, Sensitivity 68.89%, Specificity 35.29%, ROC-AUC 0.6046, PR-AUC 0.8369, Brier 0.2714.
   - Validation Split ($N=63$): Accuracy 65.08%, Sensitivity 65.22%, Specificity 64.71%, ROC-AUC 0.6419, PR-AUC 0.8600, Brier 0.2498.
3. **State C (Locally Fine-Tuned P3 Model):**  
   Our independent P3 model initialized from official weights and fine-tuned locally on the TopAneu-26 patient-level training split (Experiment `P4-P3-001`). P4-P3-001 performed partial fine-tuning of the independently implemented P3 architecture, freezing early spatial encoder stages 0–3 and optimizing deeper semantic stages 4–5 together with the relevant cross-attention/classification components (84,998,211 trainable parameters / 77.72% of total network; `only_forward_cls=True`). Full multi-task decoder fine-tuning was not performed because of the local VRAM constraint.
   - Locked Test Split ($N=62$): Accuracy **77.42%**, Sensitivity **95.56%**, Specificity 29.41%, ROC-AUC **0.8928**, PR-AUC **0.9626**, Brier **0.1294**.
   - Validation Split ($N=63$): Accuracy **88.89%**, Sensitivity **93.48%**, Specificity **76.47%**, ROC-AUC **0.9079**, PR-AUC **0.9655**, Brier **0.1104**.

---

## 3. Repository Structure

```text
brain-atlas-model/
├── README.md                           # Master project documentation
├── LICENSE                             # Project license
├── configs/                            # Declarative training and experiment configs
│   └── p4_p3_finetune.yaml             # Experiment P4-P3-001 configuration
├── src/                                # Core reusable library source code
│   ├── data/                           # Dataset, caching, and preprocessing modules
│   │   ├── dataset.py                  # TopAneuDataset with caching & 224^3 resampling
│   │   └── __init__.py
│   ├── models/                         # Neural network architectures
│   │   └── p3/                         # Independent P3 native implementation
│   │       ├── blocks.py               # 3D Residual blocks & Transposed convolutions
│   │       ├── head.py                 # Multi-head CrossAttention classification heads
│   │       ├── pooling.py              # CrossAttentionPooling module
│   │       ├── model.py                # P3Architecture (109.36M params)
│   │       └── __init__.py
│   ├── training/                       # Training, optimization, and loss engines
│   │   ├── trainer.py                  # P3Trainer supporting AMP & Grad Accumulation
│   │   └── __init__.py
│   ├── evaluation/                     # Standardization evaluation metrics
│   │   ├── evaluator.py                # P3Evaluator (AUC, Confusion Matrix, Brier)
│   │   └── __init__.py
│   └── xai/                            # Explainability hooks and analysis
├── scripts/                            # Executable CLI tools
│   ├── verify_p3_checkpoint.py         # Strict checkpoint compatibility validator
│   ├── create_split.py                 # Patient-level stratified split generator
│   ├── precache_dataset.py             # Multi-core 224^3 volume pre-cacher
│   ├── train_p3.py                     # CLI fine-tuning entrypoint
│   └── evaluate_p3.py                  # CLI evaluation and reporting entrypoint
├── experiments/                        # Persistent experiment artifacts
│   ├── splits/                         # Machine-readable split tables
│   │   ├── topaneu_v1.csv              # Patient-level Train / Val / Locked Test split
│   │   └── topaneu_v1_summary.json     # Split summary metrics & distributions
│   └── results/                        # Saved checkpoints, metrics JSONs, logs
├── reports/                            # Formal research reports and audits
│   ├── baseline/                       # Phase 1 baseline reports & predictions
│   ├── phase2/                         # Phase 2 anatomical alignment reports & tables
│   ├── phase3/                         # Phase 3 explainability & faithfulness reports
│   └── phase4/                         # Phase 4 audit, split protocol, & validation
├── references/                         # Provenance and network specifications
│   └── p3/
│       ├── PROVENANCE.md               # Authorship, citations, and SHA256 checksum
│       ├── ARCHITECTURE_SPEC.md        # Exact layer, channel, and bottleneck specs
│       └── plans.json                  # Official network configuration plans
├── tests/                              # Automated unit test suite
│   ├── test_p3_model.py                # Architecture & forward pass unit tests
│   ├── test_split.py                   # Split integrity and zero-leakage tests
│   └── test_data_pipeline.py           # Dataset tensor loading unit tests
├── topaneu_release/                    # TopAneu-26 dataset directory
│   ├── images/                         # 3D NIfTI volumes (*_0000.nii.gz)
│   ├── location_masks/                 # Multi-class aneurysm location masks
│   ├── location_jsons/                 # Case-level aneurysm annotation JSONs
│   ├── type_masks/                     # Aneurysm morphology masks
│   └── vessel_masks/                   # TopBrain vessel segmentation masks
└── scratch/                            # Local scratch and cache storage (gitignored)
    ├── cache_224/                      # Cached 224^3 preprocessed float32 tensors
    └── checkpoints/                    # Official P3 pretrained weights
```

---

## 4. Dataset & Checkpoint Locations

- **TopAneu-26 Dataset Directory:** Expected at `topaneu_release/` (415 clinical 3D volumes across 408 unique patients).
- **Pretrained P3 Checkpoint:** Expected at `scratch/checkpoints/Dataset660_26classes_resize224_4661/onlyMirror01_lr4e3_100epochs_ps224/fold_0/checkpoint_final.pth` (SHA256: `e60b539d025a8ecccf77d3ff1a45ef6888b5e059671cc8e24fff572d520fa521`).

---

## 5. Quickstart & Reproduction Commands

### 5.1 Run Automated Test Suite
```powershell
pytest tests/ -v
```

### 5.2 Verify Official Pretrained Weights
```powershell
python scripts/verify_p3_checkpoint.py --checkpoint scratch/checkpoints/.../checkpoint_final.pth
```

### 5.3 Generate / Verify Patient-Level Split
```powershell
python scripts/create_split.py --data_dir topaneu_release --seed 42
```

### 5.4 Pre-cache Preprocessed Volumes (224³)
```powershell
python scripts/precache_dataset.py --workers 4
```

### 5.5 Run Baseline Inference / Evaluation (State B)
```powershell
# Evaluate on Locked Test Split
python -u scripts/evaluate_p3.py --checkpoint scratch/checkpoints/.../checkpoint_final.pth --split test --state_name State_B_Official
```

### 5.6 Fine-tune P3 on TopAneu-26 (State C)
```powershell
python -u scripts/train_p3.py --config configs/p4_p3_finetune.yaml
```

---

## 6. Hardware Compatibility & Memory Strategy

The pipeline is engineered to run reliably on consumer/laptop GPUs (tested on **NVIDIA GeForce RTX 5050 Laptop GPU, ~8 GB VRAM**, PyTorch build `2.14.0+cu130` compiled with CUDA 13.0, NVIDIA driver 592.19 reporting CUDA 13.1 support):
- **Batch Size:** 1
- **Mixed Precision:** PyTorch Automatic Mixed Precision (`torch.amp.autocast("cuda", dtype=torch.float16)`)
- **Gradient Accumulation:** Steps = 8 (effective batch size 8)
- **Peak VRAM:** 6,568 MiB (~6.57 GB) allocated during training; ~6.55 GB during evaluation (zero host-memory paging)
- **Caching:** Pre-resampled $224^3$ float32 tensors eliminate CPU bottleneck during training.

---

## 7. Current Limitations & Scope Boundary

- **Phase 4 Scope Boundary:** Strictly encompasses P3 reproduction, validation, RSNA2025 dependency removal, patient split protocol, training pipeline, and baseline fine-tuning.
- **Excluded from Phase 4:** Dataset augmentation, augmentation ablation experiments, new anatomy-conditioned classifiers, architectural redesigns, anatomical loss constraints, and NLP integration. These are reserved for Phase 5 and beyond.
