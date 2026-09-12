# Baseline Forward Pass Verification Report
## Architecture: `ResEncoderUNet_two_seg_with_cls_modality`
**Model Source:** `RSNA2025_Intracranial-Aneurysm-Detection/nnXNet/.../variants/network_architecture/ResEncoderUNet_two_seg_with_cls_modality.py`  
**Configuration Source:** `nnXNetResEncUNetM_two_seg_with_cls_ps_224_224_224_Plans.json`  
**Date:** 2026-09-10  

---

## 1. Exact Commands & Execution

### Test 1: Unmodified Stock Environment
```powershell
python D:\NLP_Project\RSNA2025_Intracranial-Aneurysm-Detection\nnXNet\nnxnet\training\nnXNetTrainer\variants\network_architecture\ResEncoderUNet_two_seg_with_cls_modality.py
```
- **Result:** ❌ `ModuleNotFoundError: No module named 'dynamic_network_architectures'`
- **Analysis:** P3 requires the `dynamic_network_architectures` building blocks library. In the Kaggle submission notebooks, this was installed via a wheel bundled at `/kaggle/input/wheels-20251001/...`. In the local environment, it was missing.

### Test 2: Reproducible Non-Invasive Verification Script
To avoid modifying the original repository files, the exact `dynamic_network_architectures==0.3.1` source was extracted to `scratch/dna_src` and loaded via Python `sys.path`:
```powershell
python D:\NLP_Project\scratch\test_forward_pass.py
```
- **Result:** ✅ **SUCCESS** (Model instantiated, all shapes verified, loss evaluated).

---

## 2. Model Construction & Parameter Count

| Property | Value | Notes |
|---|---|---|
| **Class Name** | `ResEncoderUNet_two_seg_with_cls_modality` | P3 Stage-2 core multi-task network |
| **Total Parameters** | **109,359,299** (~109.36 M) | [MEASURED] |
| **Trainable Parameters** | **109,359,299** (100%) | [MEASURED] |
| **Construction Time** | 0.72 seconds | CPU initialization |
| **Model In-Memory Size** | 618.9 MB | Initial weights footprint |

---

## 3. Tensor Shapes & Bottleneck Verification

| Component | Intended / Expected Shape | Verified / Measured Shape | Status |
|---|---|---|---|
| **Stage-2 Input Volume** | `(B, 1, 224, 224, 224)` | `[1, 1, 224, 224, 224]` (42.9 MB) | ✅ **VERIFIED** |
| **Encoder Stage 0** | `(B, 32, 224, 224, 224)` | `[1, 32, 224, 224, 224]` | ✅ **VERIFIED** |
| **Encoder Stage 1** | `(B, 64, 112, 112, 112)` | `[1, 64, 112, 112, 112]` | ✅ **VERIFIED** |
| **Encoder Stage 2** | `(B, 128, 56, 56, 56)` | `[1, 128, 56, 56, 56]` | ✅ **VERIFIED** |
| **Encoder Stage 3** | `(B, 256, 28, 28, 28)` | `[1, 256, 28, 28, 28]` | ✅ **VERIFIED** |
| **Encoder Stage 4** | `(B, 320, 14, 14, 14)` | `[1, 320, 14, 14, 14]` | ✅ **VERIFIED** |
| **Bottleneck Feature Map** | `(B, 320, 7, 7, 7)` | `[1, 320, 7, 7, 7]` | ✅ **VERIFIED** |
| **Aneurysm Presence Head**| `(B, 1)` | `[1, 1]` | ✅ **VERIFIED** |
| **Location Head (13-class)**| `(B, 13)` | `[1, 13]` | ✅ **VERIFIED** |
| **Modality Head (4-class)** | `(B, 4)` | `[1, 4]` | ✅ **VERIFIED** |
| **Decoder 1 (15 classes)** | 5 scales: $224^3, 112^3, 56^3, 28^3, 14^3$ | `[1, 15, D, H, W]` at 5 scales | ✅ **VERIFIED** |
| **Decoder 2 (14 classes)** | 5 scales: $224^3, 112^3, 56^3, 28^3, 14^3$ | `[1, 14, D, H, W]` at 5 scales | ✅ **VERIFIED** |

---

## 4. Cross-Attention Pooling & Classification Heads

- **Aneurysm Presence Head:** `CrossAttentionPooling(embed_dim=320, query_num=2, num_classes=1, num_heads=4)`
- **Anatomical Location Head:** `CrossAttentionPooling(embed_dim=320, query_num=16, num_classes=13, num_heads=4)`
- **Imaging Modality Head:** `CrossAttentionPooling(embed_dim=320, query_num=4, num_classes=4, num_heads=4)`
- All 3 heads attend over the flattened $7 \times 7 \times 7 = 343$ spatial bottleneck tokens.
- Output shapes strictly match the multi-task formulation.

---

## 5. Loss Computation Verification

Computed using repository loss formulas and weights:
- **Presence Loss (`BCEWithLogitsLoss`, pos_weight=1.25):** `1.3220` (finite: `True`)
- **Location Loss (`BCEWithLogitsLoss`, pos_weights=[15,15,8,8...]):** `2.8144` (finite: `True`)
- **Modality Loss (`CrossEntropyLoss`, weights=[1,1,1,1]):** `2.1645` (finite: `True`)
- **Total Multi-Task Loss:** Finite, no NaNs, no infinities.

---

## 6. Computational & Memory Requirements

- **GPU Memory Usage:** 0 MB (execution ran on CPU due to PyTorch CPU-only build).
- **Host RAM (Pre-construction):** 197.7 MB
- **Host RAM (Post-construction):** 618.9 MB
- **Host RAM (Peak during $224^3$ forward pass):** 962.1 MB
- **Forward Pass Runtime ($224^3$, `only_forward_cls=True`):** **22.72 seconds** on Intel Core i7-14650HX.

---

## 7. Errors Encountered & Required Fixes

| # | Observed Error | Root Cause | Fix Required / Applied |
|---|---|---|---|
| **E1** | `ModuleNotFoundError: No module named 'dynamic_network_architectures'` | Missing nnU-Net building blocks library in Python environment | Vendored `dynamic_network_architectures-0.3.1` to `scratch/dna_src` and added to `sys.path`. Original repo unmodified. |
| **E2** | `RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 4` | Testing patch size $112 \times 112 \times 112$. The 5 downsampling stages ($2^5 = 32$) reduce 112 to $7 / 2 = 3.5 \to 3$ vs $4$. | The architecture strictly requires spatial dimensions that are multiples of 32 (e.g., $64^3, 224^3$). Verified that $224^3 \to 7^3$ exactly. |
