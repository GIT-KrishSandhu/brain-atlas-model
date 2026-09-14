# Phase 4: P3 Fine-Tuning & Comparative Benchmark Report

**Project:** Brain Atlas / Intracranial Aneurysm Research  
**Investigation Scope:** Comparative Evaluation Across Model States A, B, and C on TopAneu-26  
**Date:** September 14, 2026  
**Evaluator:** Google DeepMind / Antigravity Agentic Assistant  

---

## 1. Executive Summary & Experimental Objectives

The central objective of Phase 4 is to transition from relying on external competition implementations to an independently implemented, validated, and fine-tuned 3D neural architecture.

This report formally compares three distinct model states on the TopAneu-26 dataset:
- **State A:** Historical frozen official P3 baseline (competition weights evaluated on the full TopAneu-26 cohort, $N=415$, without local fine-tuning).
- **State B:** Our independent native PyTorch implementation (`P3Architecture`) initialized with official P3 weights, evaluated on the newly locked patient-level partitions ($N=62$).
- **State C:** Our independent P3 architecture fine-tuned on the TopAneu-26 training split (Experiment `P4-P3-001`) and evaluated on the identical locked test partition ($N=62$).

> **Important Cohort Clarification:**  
> State A is retained as the historical full-cohort P3 reference. Because State A was evaluated on $N=415$ cases while States B and C were evaluated on the locked $N=62$ test partition, State A is not used for the primary matched-cohort comparison. The primary assessment of fine-tuning is the State B versus State C comparison on the identical locked test set.

---

## 2. Evaluation Protocol & Partition Summary

Evaluations are conducted strictly under the patient-level partition protocol established in `reports/phase4/PHASE4_DATA_SPLIT_PROTOCOL.md`:
- **Validation Partition ($N=63$):** 46 positive cases, 17 negative cases (73.02% prevalence).
- **Locked Final Test Partition ($N=62$):** 45 positive cases, 17 negative cases (72.58% prevalence).
- **Zero Patient Leakage:** Multi-timepoint longitudinal series (Center-4) and multimodal CTA/MRA pairs (Center-2) are strictly grouped to avoid cross-partition contamination.
- **Threshold Policy:** Primary classification metrics are reported at the standard uncalibrated decision threshold ($\tau = 0.50$). Threshold-independent discrimination is assessed via ROC-AUC and PR-AUC.

---

## 3. Comparative Benchmark Results

### 3.1 Locked Test Split Evaluation ($N=62$)

The primary matched-cohort comparison is between State B and State C on the locked test partition ($N=62$, 45 positive, 17 negative). State A is included solely as an unmatched historical full-cohort reference.

| Metric | State A (Historical Full-Cohort Reference, Unmatched) | State B (Independent P3 + Official Weights) | State C (Fine-Tuned `P4-P3-001`) | Primary Delta (State C vs State B) |
| :--- | :---: | :---: | :---: | :---: |
| **Cohort Definition** | Full Dataset ($N=415$) | Locked Test Partition ($N=62$) | Locked Test Partition ($N=62$) | Identical Locked Cohort |
| **Accuracy** | 62.41% | 59.68% | **77.42%** | **+17.74%** |
| **Sensitivity (Recall)** | 68.42% | 68.89% | **95.56%** | **+26.67%** |
| **Specificity** | 45.95% | 35.29% | 29.41% | -5.88% |
| **Precision (PPV)** | 77.61% | 73.81% | **78.18%** | **+4.37%** |
| **Negative Predictive Value (NPV)** | 34.69% | 30.00% | **71.43%** | **+41.43%** |
| **F1-Score** | 0.7273 | 0.7126 | **0.8600** | **+0.1474** |
| **ROC-AUC** | 0.6388 | 0.6046 | **0.8928** | **+0.2882** |
| **PR-AUC (Average Precision)** | 0.8529 | 0.8369 | **0.9626** | **+0.1257** |
| **Brier Score** | 0.2491 | 0.2714 | **0.1294** | **-0.1420** |
| **Mean Inference Latency** | 392.6 ms | 1007.5 ms | 367.0 ms | -640.5 ms |

---

### 3.2 Confusion Matrices on Locked Test Set ($\tau = 0.50, N=62$)

#### State B (Frozen Official Checkpoint on Independent Code, Locked Test):
$$\begin{pmatrix} \text{TN}=6 & \text{FP}=11 \\ \text{FN}=14 & \text{TP}=31 \end{pmatrix}$$
- **True Positives (TP):** 31
- **False Negatives (FN):** 14
- **True Negatives (TN):** 6
- **False Positives (FP):** 11

#### State C (Fine-Tuned `P4-P3-001` on Independent Code, Locked Test):
$$\begin{pmatrix} \text{TN}=5 & \text{FP}=12 \\ \text{FN}=2 & \text{TP}=43 \end{pmatrix}$$
- **True Positives (TP):** 43
- **False Negatives (FN):** 2 (directly observed reduction from 14 to 2)
- **True Negatives (TN):** 5
- **False Positives (FP):** 12

---

### 3.3 Validation Split Performance ($N=63$, 46 Positive, 17 Negative)

These validation metrics were evaluated on the held-out validation partition during experiment selection and are reported separately from the locked test set:

| Metric | State B (Official Weights, Val $N=63$) | State C (Fine-Tuned `P4-P3-001`, Val $N=63$) | Delta (State C vs State B) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 65.08% | **88.89%** | **+23.81%** |
| **Sensitivity** | 65.22% | **93.48%** | **+28.26%** |
| **Specificity** | 64.71% | **76.47%** | **+11.76%** |
| **Precision** | 83.33% | **91.49%** | **+8.16%** |
| **F1-Score** | 0.7317 | **0.9247** | **+0.1930** |
| **ROC-AUC** | 0.6419 | **0.9079** | **+0.2660** |
| **PR-AUC** | 0.8600 | **0.9655** | **+0.1055** |
| **Brier Score** | 0.2498 | **0.1104** | **-0.1394** |
| **Confusion Matrix** | $\text{TP}=30, \text{FP}=6, \text{TN}=11, \text{FN}=16$ | $\text{TP}=43, \text{FP}=4, \text{TN}=13, \text{FN}=3$ | FN: $16 \rightarrow 3$ |

---

## 4. Fine-Tuning Scope & Training Run Progression (`P4-P3-001`)

### 4.1 Fine-Tuning Parameter Scope
P4-P3-001 performed partial fine-tuning of the independently implemented P3 architecture, freezing early spatial encoder stages 0–3 and optimizing deeper semantic stages 4–5 together with the relevant cross-attention/classification components. Full multi-task decoder fine-tuning was not performed because of the local VRAM constraint.

- **Total Network Parameters:** 109,359,299
- **Trainable Parameters:** 84,998,211 (77.72% of total network)
- **Frozen Parameters:** 24,361,088 (early conv encoder stages 0–3: 32, 64, 128, 256 feature channels)
- **Forward Path Configuration:** Classification-only forward path (`only_forward_cls=True`)

### 4.2 Training Progression
- **Experiment ID:** `P4-P3-001`
- **Initial Validation (before fine-tuning):** Loss = 0.8387, Acc = 65.08%, AUC = 0.6419.
- **Epoch 1:** Train Loss = 0.5657, Train Acc = 74.48%, Train AUC = 0.7794 | Val Loss = 0.4475, Val Acc = 77.78%, Val AUC = 0.8907 (duration 154.6s).
- **Epoch 2:** Train Loss = 0.4205, Train Acc = 79.31%, Train AUC = 0.8699 | Val Loss = 0.3808, Val Acc = **88.89%**, Val AUC = **0.9079** (duration 153.2s) $\rightarrow$ **Saved as `best_checkpoint.pth`**.
- **Epoch 3:** Train Loss = 0.3423, Train Acc = 84.14%, Train AUC = 0.9219 | Val Loss = 0.3921, Val Acc = 85.71%, Val AUC = 0.8926 (duration 153.3s).
- **Total Training Duration:** 461.1 seconds (~7.7 minutes).
- **Hardware Profile:** NVIDIA GeForce RTX 5050 Laptop GPU (8,150.6 MiB VRAM), PyTorch build `2.14.0+cu130` (CUDA 13.0 compiled, NVIDIA driver 592.19 reporting CUDA 13.1 support). Peak allocated VRAM was 6,568 MiB (1.35 GB headroom, zero host-memory paging).

---

## 5. Scientific Findings & Epistemic Analysis

### 5.1 OBSERVED
1. **Performance Shift on Locked Test Partition:** Fine-tuning substantially improved performance on the locked TopAneu-26 test partition relative to the independently implemented P3 initialized with official weights (ROC-AUC: 0.6046 $\rightarrow$ 0.8928, PR-AUC: 0.8369 $\rightarrow$ 0.9626, F1-Score: 0.7126 $\rightarrow$ 0.8600).
2. **Sensitivity/Specificity Trade-Off:** At the fixed $\tau=0.50$ threshold, fine-tuning substantially increased sensitivity (68.89% $\rightarrow$ 95.56%) while specificity decreased slightly (35.29% $\rightarrow$ 29.41%) on the locked test partition. False negatives directly decreased from 14 to 2, while false positives shifted from 11 to 12.
3. **Probabilistic Prediction Error:** The Brier score decreased from 0.2714 to 0.1294 on the locked test partition, indicating lower probabilistic prediction error under this metric. Formal calibration assessment was not performed in Phase 4 and remains outside the demonstrated scope.
4. **False Positive Count:** State C produced 12 false-positive predictions on the locked test partition at $\tau=0.50$.

### 5.2 INFERRED
1. **Cohort Composition & Baseline Alignment:** The difference between State A (whole dataset, $N=415$) and State B (locked test partition, $N=62$) is consistent with differences in cohort composition, sampling variation, and class prevalence between the two evaluation populations. The available numerical equivalence checks provide no observed evidence of an implementation discrepancy, although exhaustive case-level equivalence between the historical State A run and the native implementation was not performed.
2. **Distributional Adaptation:** The improvement is consistent with adaptation to distributional differences between the P3 training domain and TopAneu-26. The present experiment does not isolate scanner, intensity, demographic, acquisition, or other individual sources of domain shift.
3. **False Positive Characteristics:** Some false positives may reflect vascular structures whose imaging characteristics resemble aneurysms. Specific morphological causes of the remaining false positives have not been established by the Phase 4 experiment.

### 5.3 UNKNOWN
1. **Lesion-Level Segmentation Performance:** Dice scores for fine-tuned decoders are UNKNOWN because Phase 4 fine-tuning focused on classification heads with encoder optimization (`only_forward_cls=True`) to adhere strictly to the 8GB VRAM envelope.
2. **Multi-Center Generalization:** Generalization to external cohorts outside TopAneu-26 remains UNKNOWN until multi-institutional validation is performed in future research phases.
