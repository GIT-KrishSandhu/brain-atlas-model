# Baseline Experiment Reproducibility Log
## Brain Atlas / Intracranial Aneurysm Research Project
**Repository:** `RSNA2025_Intracranial-Aneurysm-Detection`  
**Git Base Commit:** `0ac375c` (Merge branch 'master')  
**Repository Integrity Snapshot:** Recorded in `scratch/original_repo_manifest.json` (191 files, SHA-256 verified)  
**Date:** 2026-09-10  

---

## Experiment 1: Stock Architecture Instantiation Test (B1-R)
- **Date:** 2026-09-10
- **Git Commit:** `0ac375c`
- **Command:** `python D:\NLP_Project\RSNA2025_Intracranial-Aneurysm-Detection\nnXNet\nnxnet\training\nnXNetTrainer\variants\network_architecture\ResEncoderUNet_two_seg_with_cls_modality.py`
- **Configuration:** Stock `ResEncoderUNet_two_seg_with_cls_modality.py` `__main__` block
- **Dataset Version:** N/A (synthetic tensor)
- **Number of Cases:** 0
- **Random Seed:** Default
- **Hardware:** CPU (Intel Core i7-14650HX), PyTorch 2.14.0+cpu
- **Batch Size:** 2
- **Learning Rate:** N/A
- **Epochs:** 0
- **Checkpoint:** None
- **Metrics:** N/A
- **Runtime:** < 1s
- **Errors:** `ModuleNotFoundError: No module named 'dynamic_network_architectures'`
- **Modifications:** None. Stock execution verified to fail on missing dependency.

---

## Experiment 2: P3 Architecture Construction & Forward Pass Verification (B1-M)
- **Date:** 2026-09-10
- **Git Commit:** `0ac375c`
- **Command:** `python D:\NLP_Project\scratch\test_forward_pass.py`
- **Configuration:** Exact `nnXNetResEncUNetM_two_seg_with_cls_ps_224_224_224_Plans.json` architecture:
  - 6 stages, features: `[32, 64, 128, 256, 320, 320]`
  - Kernel sizes: `[[3,3,3], ...]`
  - Strides: `[[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2]]`
  - Blocks: `[1, 3, 4, 6, 6, 6]`, decoder convs: `[1, 1, 1, 1, 1]`
  - Presence: 2 queries, 1 class; Location: 16 queries, 13 classes; Modality: 4 queries, 4 classes
  - Cross-attention: enabled (4 heads, embed_dim=320)
  - Deep supervision: enabled (5 scales per decoder)
- **Dataset Version:** Synthetic tensor `(1, 1, 224, 224, 224)`
- **Number of Cases:** 1 synthetic volume
- **Random Seed:** Torch default
- **Hardware:** CPU (Intel Core i7-14650HX), RAM: 23.6 GB
- **Batch Size:** 1
- **Learning Rate:** N/A (eval mode)
- **Epochs:** 0
- **Checkpoint:** Initialized from scratch
- **Metrics:**
  - Total Parameters: 109,359,299 (~109.36 M)
  - Bottleneck shape: `[1, 320, 7, 7, 7]` (Verified)
  - Presence logits: `[1, 1]` (Verified)
  - Location logits: `[1, 13]` (Verified)
  - Modality logits: `[1, 4]` (Verified)
  - Decoder 1 shapes: 5 scales of `[1, 15, D, H, W]` (Verified)
  - Decoder 2 shapes: 5 scales of `[1, 14, D, H, W]` (Verified)
  - Multi-task loss: Finite (Presence: 1.3220, Location: 2.8144, Modality: 2.1645)
- **Runtime:** Construction: 0.72s; $224^3$ cls forward pass: 22.72s
- **Errors:** None (with `dynamic_network_architectures` on `sys.path`)
- **Modifications:** Non-invasive vendoring of `dynamic_network_architectures-0.3.1` in `scratch/dna_src`. Original repo untouched.

---

## Experiment 3: One Batch of Real Data (B1-M)
- **Date:** 2026-09-10
- **Git Commit:** `0ac375c`
- **Command:** `python D:\NLP_Project\scratch\test_one_batch_real_data.py`
- **Configuration:** Real volume from `topaneu_release` (`topaneu_center1_mr_017`), MRA scan with aneurysm at location 28. Resampled to $224 \times 224 \times 224$, Z-score normalized (mean=290.19, std=1933.28), clipped to $[-7905.17, 7954.08]$.
- **Dataset Version:** TopAneu-26 release
- **Number of Cases:** 1 real scan
- **Random Seed:** Torch default
- **Hardware:** CPU (Intel Core i7-14650HX)
- **Batch Size:** 1
- **Learning Rate:** N/A (single step)
- **Epochs:** N/A
- **Checkpoint:** None
- **Metrics:**
  - Input tensor shape: `[1, 1, 224, 224, 224]` (NaNs: False, Infs: False)
  - Forward pass time: 79.17s
  - Forward predictions: Presence: `1.2105`, Modality: `[1.24, -0.68, 1.79, 0.42]` (all finite)
  - Multi-task loss: `1.8163` (finite)
  - Backward pass time: 137.85s
  - First conv layer gradient norm: `1.370326` (finite)
  - Classifier head gradient norm: `2.420307` (finite)
  - Peak RAM: 3964.9 MB (~3.96 GB)
- **Runtime:** ~220s total
- **Errors:** None
- **Modifications:** None to original repo. Custom runner script in `scratch/`.

---

## Experiment 4: Tiny Subset Overfit Test (B1-M)
- **Date:** 2026-09-10
- **Git Commit:** `0ac375c`
- **Command:** `python D:\NLP_Project\scratch\test_tiny_overfit.py`
- **Configuration:** Full `ResEncoderUNet_two_seg_with_cls_modality` architecture (109.36M params), optimizer AdamW (lr=1e-3, weight_decay=1e-4), loss `BCEWithLogitsLoss(pos_weight=1.25)`. Input: $64 \times 64 \times 64$ patches.
- **Dataset Version:** TopAneu-26 release
- **Number of Cases:** 4 real cases (2 positive: `topaneu_center1_mr_017`, `topaneu_center1_mr_028`; 2 negative: `topaneu_center1_mr_001`, `topaneu_center1_mr_005`)
- **Random Seed:** 42
- **Hardware:** CPU (Intel Core i7-14650HX)
- **Batch Size:** 1 (accumulated per step)
- **Learning Rate:** 1e-3
- **Epochs / Iterations:** 20 iterations
- **Checkpoint:** None
- **Metrics:**
  - Initial Loss: 0.8307
  - Final Loss: 0.6619 (20.3% loss reduction)
  - Step 1: 0.8307
  - Step 10: 0.8390
  - Step 20: 0.6619
  - Final Predicted Probabilities:
    - `topaneu_center1_mr_017` (True=1): 0.4745
    - `topaneu_center1_mr_028` (True=1): 0.4855
    - `topaneu_center1_mr_001` (True=0): 0.3093
    - `topaneu_center1_mr_005` (True=0): 0.2945
- **Runtime:** 605.90s (~10.1 min)
- **Errors:** Handled initial missing file `_002` by replacing with confirmed negative `_005`.
- **Modifications:** None to original repo.

---

## Experiment 5: Minimal 3D CNN Baseline (B0)
- **Date:** 2026-09-10
- **Git Commit:** `0ac375c`
- **Command:** `python D:\NLP_Project\scratch\run_b0_baseline.py`
- **Configuration:** 4-stage 3D CNN:
  - Conv3d(1, 16) + BN + ReLU + MaxPool(2)
  - Conv3d(16, 32) + BN + ReLU + MaxPool(2)
  - Conv3d(32, 64) + BN + ReLU + MaxPool(2)
  - Conv3d(64, 128) + BN + ReLU + AdaptiveAvgPool3d(1)
  - Linear(128, 1)
  - Parameter count: 291,585 (291.6 K)
  - Input: $64 \times 64 \times 64$
- **Dataset Version:** TopAneu-26 release
- **Number of Cases:** 24 total cases (Train: 16 cases [8 pos, 8 neg]; Val: 8 cases [4 pos, 4 neg])
- **Random Seed:** 42
- **Hardware:** CPU (Intel Core i7-14650HX)
- **Batch Size:** 4
- **Learning Rate:** 1e-3 (Adam, weight_decay=1e-4)
- **Epochs:** 10
- **Checkpoint:** Saved in `scratch/b0_results.json`
- **Metrics:**
  - Initial Train Loss: 0.8274
  - Final Train Loss: 0.6492
  - Final Val Loss: 0.7956
  - AUROC: 0.3750
  - AUPRC: 0.4833
  - Accuracy: 0.5000 (50.0%)
  - Sensitivity: 1.0000 (100.0%)
  - Specificity: 0.0000 (0.0%)
  - Precision: 0.5000 (50.0%)
  - F1 Score: 0.6667
  - Confusion Matrix: TN=0, FP=4, FN=0, TP=4
- **Runtime:** 13.17s
- **Errors:** None
- **Modifications:** None. Self-contained reference experiment.

---

## Experiment 6: CUDA Activation & GPU Baseline Verification (RTX 5050 Laptop GPU)
- **Date:** 2026-09-10
- **Git Commit:** `0ac375c`
- **Command:** `python D:\NLP_Project\scratch\gpu_baseline_test_modular.py`
- **Configuration:** P3 `ResEncoderUNet_two_seg_with_cls_modality` (109.36M params) executed on GPU (CUDA 13.0, sm_120)
- **Dataset Version:** Synthetic tensor $(1, 1, 224, 224, 224)$ + Real TopAneu scan (`topaneu_center1_mr_017`)
- **Number of Cases:** 1 synthetic volume + 1 real volume
- **Random Seed:** Default
- **Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8,150.6 MiB VRAM), PyTorch 2.14.0+cu130
- **Batch Size:** 1
- **Learning Rate:** N/A (inference & single step test)
- **Epochs:** N/A
- **Checkpoint:** None (un-checkpointed architecture test)
- **Metrics:**
  - Model construction & transfer time: **0.416s** (Weight VRAM: 452.6 MiB)
  - Synthetic $224^3$ inference time (`only_forward_cls=True`): **1.0096s (1009.6 ms)**
  - Synthetic $224^3$ inference peak VRAM: **7,355.6 MiB (7.18 GB)** $\to$ FITS IN 8GB VRAM
  - Full multi-task forward time (AMP FP16): **4.0910s** (Peak VRAM: 7,538.7 MiB / 7.36 GB)
  - Multi-task loss: `0.7299` (Presence: 0.1586, Location: 1.2964, Modality: 0.7347, all finite)
  - Backward pass time ($224^3$, AMP): **25.8470s** (Grad norm conv1: 1.9624, head: 3.7644)
  - Real TopAneu scan inference time ($224^3$): **0.8477s (847.7 ms)** (Peak VRAM: 7,387.6 MiB / 7.21 GB)
  - Real scan predicted presence probability: **0.8930**
- **Runtime:** ~35s total
- **Errors:** Handled initial FP32 dual-decoder activation memory peak by enabling AMP FP16.
- **Modifications:** Installed `torch-2.14.0+cu130` for native Blackwell sm_120 support. Original repository files unmodified.

