# P3 Architecture Specification

## 1. Overview
The P3 architecture (`ResEncoderUNet_two_seg_with_cls_modality`) is a unified multi-task 3D network coupling a deep 3D residual encoder with:
1. Multi-head cross-attention pooling for classification tasks.
2. Dual multi-scale 3D convolutional decoders for segmentation tasks.

---

## 2. Encoder Specification
- **Input Shape:** `(B, 1, 224, 224, 224)` (normalized 3D head scan).
- **Number of Stages:** 6 stages.
- **Stage Configurations:**
  - **Stage 0:** In=1, Out=32, Kernel=[3,3,3], Stride=[1,1,1], Blocks=1 (`StackedResidualBlocks` with `BasicBlockD`). Output shape: `(B, 32, 224, 224, 224)`.
  - **Stage 1:** In=32, Out=64, Kernel=[3,3,3], Stride=[2,2,2], Blocks=3. Output shape: `(B, 64, 112, 112, 112)`.
  - **Stage 2:** In=64, Out=128, Kernel=[3,3,3], Stride=[2,2,2], Blocks=4. Output shape: `(B, 128, 56, 56, 56)`.
  - **Stage 3:** In=128, Out=256, Kernel=[3,3,3], Stride=[2,2,2], Blocks=6. Output shape: `(B, 256, 28, 28, 28)`.
  - **Stage 4:** In=256, Out=320, Kernel=[3,3,3], Stride=[2,2,2], Blocks=6. Output shape: `(B, 320, 14, 14, 14)`.
  - **Stage 5 (Bottleneck):** In=320, Out=320, Kernel=[3,3,3], Stride=[2,2,2], Blocks=6. Output shape: `(B, 320, 7, 7, 7)`.
- **Normalization:** `InstanceNorm3d(affine=True, eps=1e-5)`.
- **Activation:** `LeakyReLU(negative_slope=0.01, inplace=True)`.

---

## 3. Classification Pathway
Classification heads branch directly from the Stage 5 bottleneck (`lres_input`, shape `(B, 320, 7, 7, 7)`):
- **Spatial Flattening:** `(B, 320, 7, 7, 7)` $\to$ `(B, 320, 343)` $\to$ `(343, B, 320)` (seq_len=343, batch=B, embed_dim=320).
- **CrossAttentionPooling:**
  - Learnable query tensor: `class_query` of shape `(query_num, 320)`.
  - `MultiheadAttention(embed_dim=320, num_heads=4, batch_first=False)`.
  - Output projection: `LayerNorm(320)` $\to$ Permute `(B, query_num, 320)` $\to$ Flatten `(B, query_num * 320)` $\to$ `Linear(query_num * 320, num_classes)`.
- **Head Configurations:**
  - `cls_head_list[0]` (Presence): `query_num=2`, `num_classes=1` $\implies$ Linear(640, 1).
  - `cls_head_list[1]` (Location): `query_num=16`, `num_classes=13` $\implies$ Linear(5120, 13).
  - `cls_modality_head` (Modality): `query_num=4`, `num_classes=4` $\implies$ Linear(1280, 4).

---

## 4. Decoder Pathway (Dual Decoders)
The segmentation decoders upsample from Stage 5 back to Stage 0 with skip connections from the encoder:
- **`seg_layers_1` (Vessel & Aneurysm Segmentation):** 15 output channels.
- **`seg_layers_2` (Territory Segmentation):** 14 output channels.
- **Deep Supervision:** Produces outputs at 5 resolution levels (`seg_layers_1[0..4]` and `seg_layers_2[0..4]`), with final evaluation evaluated at highest resolution (224^3).
- **Total Weights:** Exactly 583 parameter tensors summing to 109,359,299 parameters.
