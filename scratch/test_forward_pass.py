import sys, os, time, psutil
sys.path.insert(0, r'D:\NLP_Project\scratch\dna_src\dynamic_network_architectures-0.3.1')
sys.path.insert(0, r'D:\NLP_Project\RSNA2025_Intracranial-Aneurysm-Detection\nnXNet')

import torch
import torch.nn as nn
from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet_two_seg_with_cls_modality import ResEncoderUNet_two_seg_with_cls_modality

def get_ram_mb():
    p = psutil.Process()
    return p.memory_info().rss / (1024 * 1024)

print(f"Initial RAM usage: {get_ram_mb():.1f} MB")

# Instantiate model with exact plans configuration
print("Building ResEncoderUNet_two_seg_with_cls_modality...")
t0 = time.time()
model = ResEncoderUNet_two_seg_with_cls_modality(
    in_channels=1,
    out_channels_1=15,
    out_channels_2=14,
    cls_head_num_classes_list=[1, 13],
    cls_drop_out_list=[0.0, 0.0],
    cls_query_num_list=[2, 16],
    use_cross_attention=True,
    n_stages=6,
    features_per_stage=[32, 64, 128, 256, 320, 320],
    kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
    n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
    conv_bias=True,
    norm_op=nn.InstanceNorm3d,
    norm_op_kwargs={"eps": 1e-05, "affine": True},
    dropout_op=None,
    dropout_op_kwargs=None,
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={"inplace": True},
    deep_supervision=True
)
t_build = time.time() - t0

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model constructed successfully in {t_build:.2f}s!")
print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f} M)")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Post-construction RAM: {get_ram_mb():.1f} MB")

# Test with divisible by 32 shape: 64^3 (64 -> 32 -> 16 -> 8 -> 4 -> 2)
print("\n--- Test 1: Full Forward Pass on (1, 1, 64, 64, 64) ---")
model.eval()
x_64 = torch.randn(1, 1, 64, 64, 64)
with torch.no_grad():
    r1, r2, cls_preds, mod_pred = model(x_64, only_forward_cls=False)
    print(f"Presence logits shape: {cls_preds[0].shape}")
    print(f"Location logits shape: {cls_preds[1].shape}")
    print(f"Modality logits shape: {mod_pred.shape}")
    print(f"Decoder 1 outputs (scales={len(r1)}): {[list(s.shape) for s in r1]}")
    print(f"Decoder 2 outputs (scales={len(r2)}): {[list(s.shape) for s in r2]}")

# Test exact Stage-2 input: (1, 1, 224, 224, 224)
print("\n--- Test 2: Exact Stage-2 Input Shape (1, 1, 224, 224, 224) ---")
print("Allocating (1, 1, 224, 224, 224) input tensor...")
x_224 = torch.randn(1, 1, 224, 224, 224)
print(f"Input tensor size: {x_224.nelement() * 4 / (1024*1024):.1f} MB")
print(f"RAM before 224^3 forward: {get_ram_mb():.1f} MB")

# Test only_forward_cls on 224^3 (the exact production inference mode!)
print("Running only_forward_cls on 224^3...")
t_cls_start = time.time()
with torch.no_grad():
    cls_preds_224 = model(x_224, only_forward_cls=True)
t_cls = time.time() - t_cls_start

print(f"only_forward_cls completed in {t_cls:.2f}s!")
print(f"  Presence logits shape: {cls_preds_224[0].shape}")
print(f"  Location logits shape: {cls_preds_224[1].shape}")
print(f"  RAM after cls forward: {get_ram_mb():.1f} MB")

# Step through encoder to inspect bottleneck exactly
print("\nInspecting encoder stages on 224^3:")
with torch.no_grad():
    h = model.conv_encoder_blocks[0](x_224)
    print(f"  Stage 0 output: {list(h.shape)}")
    for i in range(1, len(model.conv_encoder_blocks)):
        h = model.conv_encoder_blocks[i](h)
        print(f"  Stage {i} output: {list(h.shape)}")
    bottleneck = h
    print(f"\nVerified bottleneck feature map: {list(bottleneck.shape)}")
    assert list(bottleneck.shape) == [1, 320, 7, 7, 7], f"Expected [1, 320, 7, 7, 7], got {bottleneck.shape}"
    
    cls_presence = model.cls_head_list[0](bottleneck)
    cls_location = model.cls_head_list[1](bottleneck)
    cls_modality = model.cls_modality_head(bottleneck)

print(f"Presence head output shape: {list(cls_presence.shape)} (Expected [1, 1])")
print(f"Location head output shape: {list(cls_location.shape)} (Expected [1, 13])")
print(f"Modality head output shape: {list(cls_modality.shape)} (Expected [1, 4])")

# Test Loss computation
print("\n--- Test 3: Loss Computation Verification ---")
bce_presence_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.25]))
bce_location_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15., 15., 8., 8., 8., 8., 8., 15., 15., 8., 8., 8., 8.]))
ce_modality_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0, 1.0]))

target_presence = torch.tensor([[1.0]])
target_location = torch.tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
target_modality = torch.tensor([1])

loss_presence = bce_presence_fn(cls_presence, target_presence)
loss_location = bce_location_fn(cls_location, target_location)
loss_modality = ce_modality_fn(cls_modality, target_modality)

print(f"Loss Presence: {loss_presence.item():.4f} (finite: {torch.isfinite(loss_presence).item()})")
print(f"Loss Location: {loss_location.item():.4f} (finite: {torch.isfinite(loss_location).item()})")
print(f"Loss Modality: {loss_modality.item():.4f} (finite: {torch.isfinite(loss_modality).item()})")

print("\n=== ALL FORWARD PASS AND ARCHITECTURE CHECKS PASSED ===")
