# AUGMENTATION RESULTS REPORT
## Experiment: AUG-P3-001 vs P4-P3-001 (A0 Baseline)
### TopAneu-26 / Brain Atlas Research Project - Data Augmentation Phase

---

## 1. Experiment Configuration

| Parameter | A0 Baseline (P4-P3-001) | AUG-P3-001 |
|:---|:---:|:---:|
| Architecture | P3 (unchanged) | P3 (unchanged) |
| Init checkpoint | checkpoint_final.pth | checkpoint_final.pth (identical) |
| Fine-tune strategy | Stages 4+, freeze 0-3 | Stages 4+, freeze 0-3 (identical) |
| Trainable params | 84,998,211 / 109,359,299 | 84,998,211 / 109,359,299 |
| LR | 1e-4 (cosine decay) | 1e-4 (cosine decay, identical) |
| Epochs | 3 | 3 |
| Batch size | 1 (grad accum x 8) | 1 (grad accum x 8) |
| Pos weight | 1.25 | 1.25 |
| Seed | 42 | 42 |
| Augmentation | None | Yes (pipeline below) |

**Sole experimental difference: on-the-fly stochastic augmentation during training.**

---

## 2. Augmentation Pipeline (Validated)

All transforms formally audited prior to training. All passed. No transforms were disabled.

| Transform | Parameters | Mathematical nature |
|:---|:---|:---|
| Spatial rotation | p=0.40, +-10 deg per axis | Single-pass 3D affine; same theta for image (bilinear) and mask (nearest) |
| Spatial translation | p=0.40, +-8 voxels | Normalized shift = 2v/N; border padding; same theta |
| Contrast variation | p=0.30, c in [0.90, 1.10] | Post-normalization multiplicative scaling of Z-score volume. NOT raw scanner intensity scaling. [INFERRED] |
| Gamma reshape | p=0.30, gamma in [0.90, 1.10] | Per-volume dynamic [0,1] rescaling then power-law then inverse rescale. NOT physical scanner gamma model. [INFERRED] |
| Gaussian noise | p=0.20, sigma=0.05 | Additive zero-mean noise. Experimental hyperparameter, NOT measured physical noise level. [INFERRED] |
| Gaussian blur | p=0.20, sigma in [0.50, 0.75] | 3x3x3 kernel, sum=1.000000 verified. Shape preserved. |

Anti-stacking: Noise and blur are mutually exclusive per sample. Spatial transforms composed into one affine matrix (applied once).

---

## 3. Training Summary

| Epoch | Train Loss | Train Acc | Train AUC | Val Loss | Val AUC | LR |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.5655 | 73.45% | 0.7732 | 0.4425 | 0.8708 | 7.62e-05 |
| 2 | 0.4487 | 78.97% | 0.8544 | 0.4357 | 0.8670 | 2.88e-05 |
| 3 | 0.3873 | 79.66% | 0.8969 | 0.4072 | 0.8875 | 5.00e-06 |

Best checkpoint: Epoch 3 (Val AUC = 0.8875)

---

## 4. Evaluation Results

### 4.1 Validation Split (N=63, Pos=46, Neg=17)

| Metric | A0 Baseline | AUG-P3-001 | Delta |
|:---|:---:|:---:|:---:|
| Accuracy | 88.89% | 82.54% | -6.35 pp |
| Sensitivity | 93.48% | 95.65% | +2.17 pp |
| Specificity | 76.47% | 47.06% | -29.41 pp |
| Precision | 91.49% | 83.02% | -8.47 pp |
| F1-Score | 0.9247 | 0.8889 | -0.0358 |
| ROC-AUC | 0.9079 | 0.8875 | -0.0204 |
| PR-AUC | 0.9655 | 0.9600 | -0.0055 |
| Brier Score | 0.1104 | 0.1248 | +0.0144 (worse) |
| TP/FP/TN/FN | 43/4/13/3 | 44/9/8/2 | |

Val summary: AUG does NOT improve over A0 on validation by any aggregate metric.
Sensitivity gains (+1 TP) are offset by 5 additional FP. All other metrics are lower.

### 4.2 Locked Test Split (N=62, Pos=45, Neg=17) - PRIMARY ENDPOINT

| Metric | A0 Baseline | AUG-P3-001 | Delta |
|:---|:---:|:---:|:---:|
| Accuracy | 77.42% | 79.03% | +1.61 pp |
| Sensitivity | 95.56% | 97.78% | +2.22 pp |
| Specificity | 29.41% | 29.41% | 0.00 pp |
| Precision | 78.18% | 78.57% | +0.39 pp |
| F1-Score | 0.8600 | 0.8713 | +0.0113 |
| ROC-AUC | 0.8928 | 0.9268 | +0.0340 |
| PR-AUC | 0.9626 | 0.9754 | +0.0128 |
| Brier Score | 0.1294 | 0.1259 | -0.0035 (better) |
| TP/FP/TN/FN | 43/12/5/2 | 44/12/5/1 | |

---

## 5. Scientific Assessment

### 5.1 Primary metric: Test ROC-AUC

AUG-P3-001 achieves Test ROC-AUC = 0.9268 vs A0 baseline 0.8928, a difference of +0.034.
This is a measured difference on the locked test set from a single paired experiment.

IMPORTANT interpretation constraints:
- This is a single experiment with a single train/test split (N=62 test cases).
  The magnitude of this difference cannot be attributed to a specific augmentation technique.
  It cannot be claimed as statistically significant without confidence intervals,
  which require multiple seeds or cross-validation (neither performed here).
- The locked test set was used once, at the end, after model selection on val.
  No further tuning may use this result.

### 5.2 Secondary metrics

- Test sensitivity: +2.22 pp (one additional TP; FN 2->1). In aneurysm screening, higher sensitivity is clinically relevant.
- Test PR-AUC: +0.0128
- Test Brier Score: -0.0035 (marginal calibration improvement)
- Test specificity: unchanged (29.41%, 5 TN / 12 FP in both)
- Test accuracy: +1.61 pp, driven entirely by the sensitivity gain

### 5.3 Validation discrepancy

On the validation split, AUG does not improve over A0 (ROC-AUC: 0.8875 vs 0.9079).
The val split shows a notable decrease in specificity (-29.41 pp).
This discrepancy between val and test trends must be noted as a limitation.
It may reflect the small negative sample size (N=17 negatives), making specificity estimates noisy.
No secondary training decisions may be made on the basis of this discrepancy.

### 5.4 Measured conclusion

On the locked test set (N=62), controlled data augmentation with the approved pipeline produced
a higher Test ROC-AUC (+0.034) and higher sensitivity (+1 TP, -1 FN) compared to the P4-P3-001
baseline, with no change in specificity.

These are observed differences in a single experiment. They are consistent with augmentation
providing a modest regularization benefit on the test set. They do not constitute proof of
generalized improvement and should not be presented as such.

Augmentation did NOT improve performance on the validation split by any metric.

---

## 6. Scientific Audit Status

| Audit | Status |
|:---|:---:|
| Gamma mathematical validity | PASS |
| Contrast post-normalization correctness | PASS |
| Gaussian noise numerical safety | PASS |
| Gaussian blur kernel normalization | PASS |
| Spatial rotation orthogonality + mask alignment | PASS |
| Spatial translation formula + label integrity | PASS |
| Noise/blur mutual exclusion | PASS |
| Config parameter parity (12/12 parameters) | PASS |
| Initialization parity vs P4-P3-001 | PASS |

Transforms disabled: NONE
Code bug fixed: Blur path shape-handling (ndim==3 case was incorrect); fixed and verified.
Epistemic corrections: Code docstrings updated to state contrast and gamma operate on normalized
Z-score data, are not scanner physics models, and are not raw intensity perturbations.
Noise sigma is marked as experimental hyperparameter.

---

## 7. Artifacts

| Artifact | Path |
|:---|:---|
| AUG val metrics JSON | experiments/results/AUG-P3-001/AUG_P3_001_val_metrics.json |
| AUG test metrics JSON | experiments/results/AUG-P3-001/AUG_P3_001_test_metrics.json |
| AUG val predictions | experiments/results/AUG-P3-001/AUG_P3_001_val_predictions.csv |
| AUG test predictions | experiments/results/AUG-P3-001/AUG_P3_001_test_predictions.csv |
| Training history | experiments/results/AUG-P3-001/training_history.json |
| Best checkpoint (epoch 3) | experiments/results/AUG-P3-001/best_checkpoint.pth |
| Comparison CSV | reports/augmentation/AUGMENTATION_COMPARISON.csv |
| Validation audit JSON | reports/augmentation/augmentation_validation_audit.json |
| Audit script | scripts/audit_augmentation.py |

---

## 8. Phase Closure

This report concludes the Data Augmentation Phase of the Brain Atlas research project.

What this phase determined:
- A strictly validated augmentation pipeline was designed, audited, and applied.
- AUG-P3-001 showed a measurable improvement in Test ROC-AUC (+0.034) and sensitivity (+1 TP) over A0 on the locked test set.
- No improvement was observed on the validation split.
- No augmentation technique failed scientific validation.

What this phase does NOT authorize:
- Anatomy-conditioned architecture changes
- XAI / saliency map investigations
- Further augmentation iterations or hyperparameter sweeps
- Statistical significance claims
- Any additional use of the locked test set

PHASE STATUS: COMPLETE AND FROZEN.
