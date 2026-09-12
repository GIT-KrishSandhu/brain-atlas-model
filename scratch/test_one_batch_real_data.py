import sys, os, time, psutil
sys.path.insert(0, r'D:\NLP_Project\scratch\dna_src\dynamic_network_architectures-0.3.1')
sys.path.insert(0, r'D:\NLP_Project\RSNA2025_Intracranial-Aneurysm-Detection\nnXNet')

import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import json
from scipy.ndimage import zoom

from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet_two_seg_with_cls_modality import ResEncoderUNet_two_seg_with_cls_modality

def get_ram_mb():
    p = psutil.Process()
    return p.memory_info().rss / (1024 * 1024)

print(f"Initial RAM: {get_ram_mb():.1f} MB")

# 1. Load real scan from topaneu_release
case_id = 'topaneu_center1_mr_017'
img_path = rf'D:\NLP_Project\topaneu_release\images\{case_id}_0000.nii.gz'
json_path = rf'D:\NLP_Project\topaneu_release\location_jsons\{case_id}.json'

print(f"Loading real scan: {img_path}")
nii = nib.load(img_path)
raw_img = nii.get_fdata().astype(np.float32)
print(f"Raw image shape: {raw_img.shape}, min: {raw_img.min():.1f}, max: {raw_img.max():.1f}")

with open(json_path) as fp:
    loc_data = json.load(fp)
locations = loc_data.get('locations', [])
print(f"Aneurysm locations in {case_id}: {locations} (positive: {len(locations) > 0})")

# 2. Preprocess: resample / crop to 224x224x224 and apply Z-score normalization
print("Preprocessing real volume to 224x224x224...")
factors = [224.0 / s for s in raw_img.shape]
resampled_img = zoom(raw_img, factors, order=1)
print(f"Resampled shape: {resampled_img.shape}")

# Z-score normalization using P3 plans parameters (mean=290.19, std=1933.28)
# and clip
norm_img = (resampled_img - 290.1886) / 1933.2790
norm_img = np.clip(norm_img, -7905.17, 7954.08)

# Convert to torch tensor (B=1, C=1, 224, 224, 224)
x_real = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float()
print(f"Input tensor shape: {x_real.shape}, dtype: {x_real.dtype}")
print(f"Check NaNs/Infs in input: NaNs={torch.isnan(x_real).any().item()}, Infs={torch.isinf(x_real).any().item()}")

# Target labels
# Presence: 1.0
target_presence = torch.tensor([[1.0 if len(locations) > 0 else 0.0]], dtype=torch.float32)
# 13-location multi-label: for demonstration, index 0 set to 1.0 if positive
target_location = torch.zeros((1, 13), dtype=torch.float32)
if len(locations) > 0:
    target_location[0, 0] = 1.0
# Modality: center1_mr is MRA -> class index 1 (0: CTA, 1: MRA, 2: T2, 3: T1-post)
target_modality = torch.tensor([1], dtype=torch.long)

print(f"Target Presence: {target_presence}")
print(f"Target Location: {target_location}")
print(f"Target Modality: {target_modality}")

# 3. Model construction
print("Instantiating ResEncoderUNet_two_seg_with_cls_modality...")
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

# 4. Forward pass in training mode (only_forward_cls to fit into RAM safely)
model.train()
print("Running forward pass (only_forward_cls=True) with gradients enabled...")
t0 = time.time()
cls_preds = model(x_real, only_forward_cls=True)
presence_pred, location_pred = cls_preds[0], cls_preds[1]

# Also run modality head on bottleneck
h = model.conv_encoder_blocks[0](x_real)
for i in range(1, len(model.conv_encoder_blocks)):
    h = model.conv_encoder_blocks[i](h)
modality_pred = model.cls_modality_head(h)
t_fwd = time.time() - t0
print(f"Forward pass completed in {t_fwd:.2f}s!")

print(f"Presence prediction: {presence_pred.detach().numpy()}, finite: {torch.isfinite(presence_pred).all().item()}")
print(f"Location prediction: {location_pred.detach().numpy()}, finite: {torch.isfinite(location_pred).all().item()}")
print(f"Modality prediction: {modality_pred.detach().numpy()}, finite: {torch.isfinite(modality_pred).all().item()}")

# 5. Compute losses
bce_presence_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.25]))
bce_location_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15., 15., 8., 8., 8., 8., 8., 15., 15., 8., 8., 8., 8.]))
ce_modality_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0, 1.0]))

loss_p = bce_presence_fn(presence_pred, target_presence)
loss_loc = bce_location_fn(location_pred, target_location)
loss_mod = ce_modality_fn(modality_pred, target_modality)

total_cls_loss = (loss_p + loss_loc + loss_mod) / 3.0

print(f"Loss Presence: {loss_p.item():.4f}")
print(f"Loss Location: {loss_loc.item():.4f}")
print(f"Loss Modality: {loss_mod.item():.4f}")
print(f"Total Loss: {total_cls_loss.item():.4f} (finite: {torch.isfinite(total_cls_loss).item()})")

# 6. Backward pass test
print("Running backward pass...")
t_bwd0 = time.time()
total_cls_loss.backward()
t_bwd = time.time() - t_bwd0
print(f"Backward pass succeeded in {t_bwd:.2f}s!")

# Check gradients
first_conv_grad = model.conv_encoder_blocks[0].blocks[0].conv1.conv.weight.grad
cls_head_grad = model.cls_head_list[0].pooling.classifier.weight.grad

print(f"First conv layer grad norm: {first_conv_grad.norm().item():.6f}")
print(f"Classifier head grad norm: {cls_head_grad.norm().item():.6f}")
print(f"Grads finite: {torch.isfinite(first_conv_grad).all().item()} and {torch.isfinite(cls_head_grad).all().item()}")
print(f"Peak RAM: {get_ram_mb():.1f} MB")

print("\n=== ONE BATCH REAL DATA TEST COMPLETED SUCCESSFULLY ===")
