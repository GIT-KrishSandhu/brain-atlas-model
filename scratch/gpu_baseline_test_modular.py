import sys, os, time, gc, json
sys.path.insert(0, r'D:\NLP_Project\scratch\dna_src\dynamic_network_architectures-0.3.1')
sys.path.insert(0, r'D:\NLP_Project\RSNA2025_Intracranial-Aneurysm-Detection\nnXNet')

import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet_two_seg_with_cls_modality import ResEncoderUNet_two_seg_with_cls_modality

def get_vram_mb():
    return torch.cuda.memory_allocated() / (1024 * 1024)

def get_max_vram_mb():
    return torch.cuda.max_memory_allocated() / (1024 * 1024)

report = {}

print("================================================================")
print("PART 1: GPU & CUDA VERIFICATION (RTX 5050)")
print("================================================================")
cuda_avail = torch.cuda.is_available()
dev_count = torch.cuda.device_count()
dev_name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
total_vram = props.total_memory / (1024 * 1024)
cc = f"{props.major}.{props.minor}"
arch_list = torch.cuda.get_arch_list()

print(f"torch.cuda.is_available(): {cuda_avail}")
print(f"Device Name: {dev_name}")
print(f"Total VRAM: {total_vram:.1f} MiB ({total_vram/1024:.2f} GB)")
print(f"Compute Capability: sm_{props.major}{props.minor} (CC {cc})")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Architecture List: {arch_list}")
print(f"sm_120 Supported: {'sm_120' in arch_list}")

# Quick warmup
x = torch.randn(100, 100, device='cuda')
_ = x @ x.T
del x
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

report['gpu_model'] = dev_name
report['total_vram_mb'] = round(total_vram, 1)
report['compute_capability'] = cc
report['torch_version'] = torch.__version__
report['sm_120_supported'] = 'sm_120' in arch_list

print("\n================================================================")
print("PART 2: MODEL CONSTRUCTION (ResEncoderUNet_two_seg_with_cls_modality)")
print("================================================================")
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
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={"inplace": True},
    deep_supervision=True
).cuda()

torch.cuda.synchronize()
t_build = time.time() - t0
model_params = sum(p.numel() for p in model.parameters())
weight_vram = get_vram_mb()

print(f"Construction & GPU transfer time: {t_build:.3f}s")
print(f"Total Parameters: {model_params:,} ({model_params/1e6:.2f} M)")
print(f"VRAM occupied by weights: {weight_vram:.1f} MiB")

report['construction_time_s'] = round(t_build, 3)
report['parameter_count'] = model_params
report['weight_vram_mb'] = round(weight_vram, 1)

print("\n================================================================")
print("PART 3: P3 INFERENCE FORWARD PASS (only_forward_cls=True)")
print("================================================================")
# Input = (1, 1, 224, 224, 224)
x_syn = torch.randn(1, 1, 224, 224, 224, device='cuda')
torch.cuda.synchronize()
input_mem = get_vram_mb() - weight_vram
print(f"Input shape: {list(x_syn.shape)}")
print(f"Input VRAM: {input_mem:.1f} MiB")

model.eval()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

torch.cuda.synchronize()
t_inf0 = time.time()
with torch.no_grad():
    cls_preds = model(x_syn, only_forward_cls=True)
torch.cuda.synchronize()
t_inf = time.time() - t_inf0
inf_peak_vram = get_max_vram_mb()

print(f"Inference forward pass time: {t_inf:.4f}s ({t_inf*1000:.1f} ms)!")
print(f"Peak VRAM during inference: {inf_peak_vram:.1f} MiB ({inf_peak_vram/1024:.2f} GB)")
print(f"Presence logits shape: {list(cls_preds[0].shape)} (Expected [1, 1])")
print(f"Location logits shape: {list(cls_preds[1].shape)} (Expected [1, 13])")
print(f"Presence logit value: {cls_preds[0].item():.4f}, prob: {torch.sigmoid(cls_preds[0]).item():.4f}")

report['inference_input_shape'] = [1, 1, 224, 224, 224]
report['inference_time_ms'] = round(t_inf * 1000, 1)
report['inference_peak_vram_mb'] = round(inf_peak_vram, 1)
report['presence_shape'] = list(cls_preds[0].shape)
report['location_shape'] = list(cls_preds[1].shape)

print("\n================================================================")
print("PART 4: FULL MULTI-TASK FORWARD PASS ON 224^3 (AMP FP16)")
print("================================================================")
# Test full multi-task (dual decoders + cls + modality) with AMP (FP16)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

try:
    torch.cuda.synchronize()
    t_full0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            r1, r2, full_cls, mod_pred = model(x_syn, only_forward_cls=False)
    torch.cuda.synchronize()
    t_full = time.time() - t_full0
    full_peak_vram = get_max_vram_mb()
    
    print(f"Full Multi-Task Forward (AMP) Time: {t_full:.4f}s ({t_full*1000:.1f} ms)!")
    print(f"Peak VRAM: {full_peak_vram:.1f} MiB ({full_peak_vram/1024:.2f} GB)")
    print(f"Modality shape: {list(mod_pred.shape)} (Expected [1, 4])")
    print(f"Decoder 1 (15 classes) shapes: {[list(s.shape) for s in r1]}")
    print(f"Decoder 2 (14 classes) shapes: {[list(s.shape) for s in r2]}")
    
    report['full_fwd_amp_time_ms'] = round(t_full * 1000, 1)
    report['full_fwd_amp_peak_vram_mb'] = round(full_peak_vram, 1)
    report['modality_shape'] = list(mod_pred.shape)
    report['dec1_shapes'] = [list(s.shape) for s in r1]
    report['dec2_shapes'] = [list(s.shape) for s in r2]
except Exception as e:
    print(f"Full 224^3 forward error: {e}")
    report['full_fwd_224_error'] = str(e)

print("\n================================================================")
print("PART 5: LOSS COMPUTATION ON GPU")
print("================================================================")
bce_p_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.25], device='cuda'))
bce_loc_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15., 15., 8., 8., 8., 8., 8., 15., 15., 8., 8., 8., 8.], device='cuda'))
ce_mod_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0, 1.0], device='cuda'))

target_p = torch.tensor([[1.0]], device='cuda')
target_loc = torch.zeros((1, 13), device='cuda')
target_loc[0, 2] = 1.0
target_mod = torch.tensor([1], device='cuda')

loss_p = bce_p_fn(cls_preds[0], target_p)
loss_loc = bce_loc_fn(cls_preds[1], target_loc)
loss_mod = ce_mod_fn(mod_pred.float(), target_mod)
total_loss = (loss_p + loss_loc + loss_mod) / 3.0

print(f"Loss Presence: {loss_p.item():.4f} (finite: {torch.isfinite(loss_p).item()})")
print(f"Loss Location: {loss_loc.item():.4f} (finite: {torch.isfinite(loss_loc).item()})")
print(f"Loss Modality: {loss_mod.item():.4f} (finite: {torch.isfinite(loss_mod).item()})")
print(f"Total Multi-Task Loss: {total_loss.item():.4f} (finite: {torch.isfinite(total_loss).item()})")

report['loss_presence'] = round(loss_p.item(), 4)
report['loss_location'] = round(loss_loc.item(), 4)
report['loss_modality'] = round(loss_mod.item(), 4)
report['total_loss'] = round(total_loss.item(), 4)

print("\n================================================================")
print("PART 6: BACKWARD PASS ON GPU (224^3 Input)")
print("================================================================")
model.train()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Train forward + backward with AMP on 224^3
torch.cuda.synchronize()
t_bwd_fwd0 = time.time()
with torch.amp.autocast('cuda', dtype=torch.float16):
    train_cls = model(x_syn, only_forward_cls=True)
    loss_train = bce_p_fn(train_cls[0], target_p)

torch.cuda.synchronize()
t_train_fwd = time.time() - t_bwd_fwd0

t_bwd0 = time.time()
loss_train.backward()
torch.cuda.synchronize()
t_bwd = time.time() - t_bwd0
bwd_peak_vram = get_max_vram_mb()

conv1_grad = model.conv_encoder_blocks[0].blocks[0].conv1.conv.weight.grad
head_grad = model.cls_head_list[0].pooling.classifier.weight.grad

print(f"Train mode forward time (AMP): {t_train_fwd:.4f}s ({t_train_fwd*1000:.1f} ms)")
print(f"Backward pass time (AMP): {t_bwd:.4f}s ({t_bwd*1000:.1f} ms)!")
print(f"Peak VRAM during forward + backward: {bwd_peak_vram:.1f} MiB ({bwd_peak_vram/1024:.2f} GB)")
print(f"First Conv Layer Gradient Norm: {conv1_grad.norm().item():.6f} (finite: {torch.isfinite(conv1_grad).all().item()})")
print(f"Cls Head Gradient Norm: {head_grad.norm().item():.6f} (finite: {torch.isfinite(head_grad).all().item()})")

report['train_fwd_time_ms'] = round(t_train_fwd * 1000, 1)
report['backward_time_ms'] = round(t_bwd * 1000, 1)
report['backward_peak_vram_mb'] = round(bwd_peak_vram, 1)
report['conv1_grad_norm'] = round(conv1_grad.norm().item(), 6)
report['head_grad_norm'] = round(head_grad.norm().item(), 6)

# Cleanup synthetic tensors
del x_syn, cls_preds, r1, r2, full_cls, mod_pred, train_cls, loss_train
model.zero_grad(set_to_none=True)
torch.cuda.empty_cache()

print("\n================================================================")
print("PART 7: REAL TOPANEU SCAN GPU INFERENCE (topaneu_center1_mr_017)")
print("================================================================")
case_id = 'topaneu_center1_mr_017'
img_path = rf'D:\NLP_Project\topaneu_release\images\{case_id}_0000.nii.gz'
json_path = rf'D:\NLP_Project\topaneu_release\location_jsons\{case_id}.json'

print(f"Loading real scan: {img_path}")
nii = nib.load(img_path)
raw_scan = nii.get_fdata().astype(np.float32)
with open(json_path) as fp:
    meta = json.load(fp)
gt_locations = meta.get('locations', [])

print(f"Raw dimensions: {raw_scan.shape}, Spacing: {nii.header.get_zooms()}")
print(f"Ground Truth Aneurysm Locations: {gt_locations} (Positive: {len(gt_locations) > 0})")

# Resample to 224x224x224
t_prep0 = time.time()
factors = [224.0 / s for s in raw_scan.shape]
resampled = zoom(raw_scan, factors, order=1)
# P3 Z-score normalization
normalized = (resampled - 290.1886) / 1933.2790
normalized = np.clip(normalized, -7905.17, 7954.08)
t_prep = time.time() - t_prep0
print(f"Preprocessing completed in {t_prep:.2f}s")

# Transfer to GPU
x_real_gpu = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float().cuda()
print(f"GPU tensor shape: {list(x_real_gpu.shape)}, device: {x_real_gpu.device}")

model.eval()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

torch.cuda.synchronize()
t_real0 = time.time()
with torch.no_grad():
    real_out = model(x_real_gpu, only_forward_cls=True)
torch.cuda.synchronize()
t_real = time.time() - t_real0
real_vram = get_max_vram_mb()

real_presence_p = torch.sigmoid(real_out[0]).item()
real_loc_probs = torch.sigmoid(real_out[1])[0].cpu().numpy().tolist()

print(f"\nREAL GPU INFERENCE RESULTS:")
print(f"Inference Time: {t_real:.4f}s ({t_real*1000:.1f} ms)!")
print(f"Peak VRAM: {real_vram:.1f} MiB ({real_vram/1024:.2f} GB)")
print(f"Predicted Presence Probability: {real_presence_p:.4f}")
print(f"Ground Truth Presence: 1.0 (Positive)")
print(f"Top 3 Predicted Location Indices: {np.argsort(real_loc_probs)[-3:][::-1]}")

report['real_case_id'] = case_id
report['real_raw_shape'] = list(raw_scan.shape)
report['real_gt_locations'] = gt_locations
report['real_inference_time_ms'] = round(t_real * 1000, 1)
report['real_peak_vram_mb'] = round(real_vram, 1)
report['real_predicted_presence_prob'] = round(real_presence_p, 4)

# Save JSON results
with open(r'D:\NLP_Project\scratch\gpu_baseline_final_results.json', 'w') as fp:
    json.dump(report, fp, indent=2)

print("\n=== ALL GPU BASELINE TESTS EXECUTED AND SAVED SUCCESSFULLY ===")
