import sys, os, time, gc
sys.path.insert(0, r'D:\NLP_Project\scratch\dna_src\dynamic_network_architectures-0.3.1')
sys.path.insert(0, r'D:\NLP_Project\RSNA2025_Intracranial-Aneurysm-Detection\nnXNet')

import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import json
from scipy.ndimage import zoom

from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet_two_seg_with_cls_modality import ResEncoderUNet_two_seg_with_cls_modality

def get_vram_mb():
    return torch.cuda.memory_allocated() / (1024 * 1024)

def get_max_vram_mb():
    return torch.cuda.max_memory_allocated() / (1024 * 1024)

print("=== PART 1: CUDA & GPU ENVIRONMENT VERIFICATION ===")
print("torch.__version__:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
device_name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
total_vram_mb = props.total_memory / (1024 * 1024)
cc = f"{props.major}.{props.minor}"
arch_list = torch.cuda.get_arch_list()

print(f"GPU Model: {device_name}")
print(f"Total VRAM: {total_vram_mb:.1f} MiB ({total_vram_mb/1024:.2f} GB)")
print(f"Compute Capability: sm_{props.major}{props.minor} ({cc})")
print(f"Supported Arch List: {arch_list}")
print(f"Native sm_120 Support: {'sm_120' in arch_list}")

# Quick warmup
torch.cuda.reset_peak_memory_stats()
x_test = torch.randn(100, 100, device='cuda')
_ = x_test @ x_test.T
del x_test
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("\n=== PART 2: P3 MODEL CONSTRUCTION ON GPU ===")
print(f"VRAM before model construction: {get_vram_mb():.1f} MiB")
t_build0 = time.time()

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
).cuda()

torch.cuda.synchronize()
t_build = time.time() - t_build0
model_params = sum(p.numel() for p in model.parameters())
vram_after_build = get_vram_mb()

print(f"P3 Model constructed & moved to CUDA in: {t_build:.3f}s")
print(f"Total Parameters: {model_params:,} ({model_params/1e6:.2f} M)")
print(f"VRAM occupied by model weights: {vram_after_build:.1f} MiB")

print("\n=== PART 3: FORWARD PASS WITH SYNTHETIC INPUT (1, 1, 224, 224, 224) ===")
# Allocate input on GPU
t_alloc0 = time.time()
x_syn = torch.randn(1, 1, 224, 224, 224, device='cuda')
torch.cuda.synchronize()
input_vram = get_vram_mb() - vram_after_build
print(f"Input tensor shape: {list(x_syn.shape)}")
print(f"Input memory on GPU: {input_vram:.1f} MiB")

# Test 3A: P3 Primary Inference Mode (only_forward_cls=True)
model.eval()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("\n--- Test 3A: GPU Inference Mode (only_forward_cls=True) ---")
torch.cuda.synchronize()
t_fwd_cls0 = time.time()
with torch.no_grad():
    cls_out = model(x_syn, only_forward_cls=True)
torch.cuda.synchronize()
t_fwd_cls = time.time() - t_fwd_cls0
peak_vram_cls = get_max_vram_mb()

print(f"Inference forward pass time: {t_fwd_cls:.4f}s ({t_fwd_cls*1000:.1f} ms)!")
print(f"Presence logits shape: {list(cls_out[0].shape)} (Expected [1, 1])")
print(f"Location logits shape: {list(cls_out[1].shape)} (Expected [1, 13])")
print(f"Peak VRAM during inference: {peak_vram_cls:.1f} MiB ({peak_vram_cls/1024:.2f} GB)")

# Test 3B: Full Multi-Task Forward Pass (seg + cls + modality)
print("\n--- Test 3B: Full Multi-Task Forward Pass (Seg + Cls + Modality) ---")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

torch.cuda.synchronize()
t_fwd_full0 = time.time()
with torch.no_grad():
    r1, r2, cls_preds, mod_pred = model(x_syn, only_forward_cls=False)
torch.cuda.synchronize()
t_fwd_full = time.time() - t_fwd_full0
peak_vram_full = get_max_vram_mb()

print(f"Full multi-task forward pass time: {t_fwd_full:.4f}s ({t_fwd_full*1000:.1f} ms)!")
print(f"Peak VRAM during full forward pass: {peak_vram_full:.1f} MiB ({peak_vram_full/1024:.2f} GB)")
print(f"Presence output shape: {list(cls_preds[0].shape)}")
print(f"Location output shape: {list(cls_preds[1].shape)}")
print(f"Modality output shape: {list(mod_pred.shape)}")
print(f"Decoder 1 (15 classes) shapes: {[list(s.shape) for s in r1]}")
print(f"Decoder 2 (14 classes) shapes: {[list(s.shape) for s in r2]}")

print("\n=== PART 4: LOSS COMPUTATION & BACKWARD PASS ON GPU ===")
# Loss functions on GPU
bce_presence_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.25], device='cuda'))
bce_location_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15., 15., 8., 8., 8., 8., 8., 15., 15., 8., 8., 8., 8.], device='cuda'))
ce_modality_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0, 1.0], device='cuda'))

target_p = torch.tensor([[1.0]], device='cuda')
target_loc = torch.zeros((1, 13), device='cuda')
target_loc[0, 1] = 1.0 # Basilar Tip
target_mod = torch.tensor([1], device='cuda') # MRA

# Compute loss
loss_p = bce_presence_fn(cls_preds[0], target_p)
loss_loc = bce_location_fn(cls_preds[1], target_loc)
loss_mod = ce_modality_fn(mod_pred, target_mod)
total_loss = (loss_p + loss_loc + loss_mod) / 3.0

print(f"Loss Presence: {loss_p.item():.4f} (finite: {torch.isfinite(loss_p).item()})")
print(f"Loss Location: {loss_loc.item():.4f} (finite: {torch.isfinite(loss_loc).item()})")
print(f"Loss Modality: {loss_mod.item():.4f} (finite: {torch.isfinite(loss_mod).item()})")
print(f"Total Loss: {total_loss.item():.4f} (finite: {torch.isfinite(total_loss).item()})")

# Test Backward Pass with autocast (FP16/BF16) and gradient tracking
print("\n--- Test 4B: Backward Pass on GPU (with AMP Autocast) ---")
model.train()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Forward in train mode with AMP to fit in 8GB VRAM
t_train_fwd0 = time.time()
with torch.amp.autocast('cuda', dtype=torch.float16):
    cls_train_out = model(x_syn, only_forward_cls=True)
    loss_train = bce_presence_fn(cls_train_out[0], target_p)

torch.cuda.synchronize()
t_train_fwd = time.time() - t_train_fwd0

t_bwd0 = time.time()
loss_train.backward()
torch.cuda.synchronize()
t_bwd = time.time() - t_bwd0
peak_vram_train = get_max_vram_mb()

print(f"Train mode forward time (AMP): {t_train_fwd:.4f}s")
print(f"Backward pass time (AMP): {t_bwd:.4f}s")
print(f"Peak VRAM during forward + backward: {peak_vram_train:.1f} MiB ({peak_vram_train/1024:.2f} GB)")

first_conv_grad = model.conv_encoder_blocks[0].blocks[0].conv1.conv.weight.grad
cls_head_grad = model.cls_head_list[0].pooling.classifier.weight.grad
print(f"First conv gradient norm: {first_conv_grad.norm().item():.6f} (finite: {torch.isfinite(first_conv_grad).all().item()})")
print(f"Cls head gradient norm: {cls_head_grad.norm().item():.6f} (finite: {torch.isfinite(cls_head_grad).all().item()})")

# Clean up synthetic tensors
del x_syn, cls_out, r1, r2, cls_preds, mod_pred, cls_train_out, loss_train
model.zero_grad(set_to_none=True)
torch.cuda.empty_cache()

print("\n=== PART 5: REAL TOPANEU CASE THROUGH PREPROCESSING & P3 MODEL ON GPU ===")
case_id = 'topaneu_center1_mr_017'
img_path = rf'D:\NLP_Project\topaneu_release\images\{case_id}_0000.nii.gz'
json_path = rf'D:\NLP_Project\topaneu_release\location_jsons\{case_id}.json'

print(f"Loading real scan: {img_path}")
t_load0 = time.time()
nii = nib.load(img_path)
raw_data = nii.get_fdata().astype(np.float32)
with open(json_path) as fp:
    loc_data = json.load(fp)
locations = loc_data.get('locations', [])

# Resample to 224x224x224
factors = [224.0 / s for s in raw_data.shape]
resampled_data = zoom(raw_data, factors, order=1)
# P3 Z-score normalization
norm_data = (resampled_data - 290.1886) / 1933.2790
norm_data = np.clip(norm_data, -7905.17, 7954.08)
t_prep = time.time() - t_load0
print(f"Preprocessing (load + zoom to 224^3 + norm) completed in: {t_prep:.2f}s")

# Transfer to GPU
x_real_gpu = torch.from_numpy(norm_data).unsqueeze(0).unsqueeze(0).float().cuda()
print(f"Real scan GPU tensor: shape={list(x_real_gpu.shape)}, device={x_real_gpu.device}")

model.eval()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Real GPU inference
torch.cuda.synchronize()
t_real_inf0 = time.time()
with torch.no_grad():
    real_cls_out = model(x_real_gpu, only_forward_cls=True)
torch.cuda.synchronize()
t_real_inf = time.time() - t_real_inf0
real_peak_vram = get_max_vram_mb()

real_presence_logit = real_cls_out[0].item()
real_presence_prob = torch.sigmoid(real_cls_out[0]).item()
real_location_probs = torch.sigmoid(real_cls_out[1])[0].cpu().numpy().tolist()

print(f"\nReal Case GPU Inference Completed in: {t_real_inf:.4f}s ({t_real_inf*1000:.1f} ms)!")
print(f"Peak VRAM during real scan inference: {real_peak_vram:.1f} MiB ({real_peak_vram/1024:.2f} GB)")
print(f"Ground Truth: Aneurysm Positive at Location(s) {locations}")
print(f"Predicted Presence Logit: {real_presence_logit:.4f}")
print(f"Predicted Presence Probability: {real_presence_prob:.4f}")
print(f"Top-3 Predicted Locations (indices): {np.argsort(real_location_probs)[-3:][::-1]}")

results_summary = {
    "gpu_name": device_name,
    "total_vram_mb": round(total_vram_mb, 1),
    "compute_capability": cc,
    "torch_version": torch.__version__,
    "cuda_arch_list": arch_list,
    "model_params": model_params,
    "model_vram_mb": round(vram_after_build, 1),
    "syn_inf_time_s": round(t_fwd_cls, 4),
    "syn_inf_peak_vram_mb": round(peak_vram_cls, 1),
    "syn_full_fwd_time_s": round(t_fwd_full, 4),
    "syn_full_peak_vram_mb": round(peak_vram_full, 1),
    "train_amp_bwd_time_s": round(t_bwd, 4),
    "train_amp_peak_vram_mb": round(peak_vram_train, 1),
    "real_inf_time_s": round(t_real_inf, 4),
    "real_peak_vram_mb": round(real_peak_vram, 1),
    "real_presence_prob": round(real_presence_prob, 4),
    "case_id": case_id
}

with open(r'D:\NLP_Project\scratch\gpu_results_summary.json', 'w') as fp:
    json.dump(results_summary, fp, indent=2)

print("\n=== ALL GPU BASELINE TESTS COMPLETED SUCCESSFULLY ===")
