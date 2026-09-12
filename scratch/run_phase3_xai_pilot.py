import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import pandas as pd

# Configure module paths
sys.path.insert(0, r'D:\NLP_Project\scratch\dna_src\dynamic_network_architectures-0.3.1')
sys.path.insert(0, r'D:\NLP_Project\RSNA2025_Intracranial-Aneurysm-Detection\nnXNet')

from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet_two_seg_with_cls_modality import (
    ResEncoderUNet_two_seg_with_cls_modality
)

TOPANEU_DIR = r"D:\NLP_Project\topaneu_release"
STAGE2_CKPT = r"D:\NLP_Project\scratch\checkpoints\Dataset660_26classes_resize224_4661\onlyMirror01_lr4e3_100epochs_ps224\fold_0\checkpoint_final.pth"
PRED_DIR = r"D:\NLP_Project\scratch\phase2_segmentation_predictions"
OUT_PILOT_DIR = r"D:\NLP_Project\scratch\phase3_pilot_outputs"
os.makedirs(OUT_PILOT_DIR, exist_ok=True)

# Pilot cohort representing key clinical/prediction strata
PILOT_CASES = [
    {"case_id": "topaneu_center1_mr_056", "category": "TP (Correct Location)"},
    {"case_id": "topaneu_center1_mr_148", "category": "TP (Wrong Location / High Seg Dice)"},
    {"case_id": "topaneu_center1_mr_017", "category": "FN (Vessel-Aligned Parent)"},
    {"case_id": "topaneu_center1_mr_028", "category": "FN (Saccular Aneurysm)"},
    {"case_id": "topaneu_center1_mr_018", "category": "FP (False Alarm)"},
    {"case_id": "topaneu_center1_mr_001", "category": "TN (Healthy Control)"}
]

def load_p3_model(device):
    print("Loading ResEncoderUNet_two_seg_with_cls_modality...")
    model = ResEncoderUNet_two_seg_with_cls_modality(
        in_channels=1, out_channels_1=15, out_channels_2=14,
        cls_head_num_classes_list=[1, 13], cls_drop_out_list=[0.0, 0.0],
        cls_query_num_list=[2, 16], use_cross_attention=True,
        n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320],
        kernel_sizes=[[3, 3, 3]] * 6, strides=[[1, 1, 1]] + [[2, 2, 2]] * 5,
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6], n_conv_per_stage_decoder=[1] * 5,
        conv_bias=True, norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        dropout_op=None, dropout_op_kwargs=None, nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True}, deep_supervision=False
    ).to(device)
    
    ckpt = torch.load(STAGE2_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["network_weights"], strict=True)
    model.eval()
    print("Pretrained weights loaded strictly.")
    return model

def normalize_attribution(attr_tensor):
    """Normalize 3D attribution tensor to [0, 1] range."""
    attr = attr_tensor.squeeze().detach()
    a_min = attr.min()
    a_max = attr.max()
    if a_max - a_min > 1e-8:
        norm_attr = (attr - a_min) / (a_max - a_min)
    else:
        norm_attr = torch.zeros_like(attr)
    return norm_attr

def compute_attribution_methods(model, vol_norm, device):
    """
    Computes:
    1. 3D Grad-CAM
    2. 3D Grad-CAM++
    3. Cross-Attention Weights
    4. Layer Integrated Gradients (Bottleneck IG, m=20)
    """
    results = {}
    
    # 1. Forward pass through encoder under no_grad to get bottleneck
    with torch.no_grad():
        conv_enc_outputs = [model.conv_encoder_blocks[0](vol_norm)]
        for i in range(1, len(model.conv_encoder_blocks)):
            conv_enc_outputs.append(model.conv_encoder_blocks[i](conv_enc_outputs[-1]))
        bottleneck_act = conv_enc_outputs[-1] # shape (1, 320, 7, 7, 7)
        
    # Hook for Cross-Attention weights
    attn_dict = {}
    def hook_attn(module, inp, out):
        attn_dict['weights'] = out[1] # shape (1, 2, 343)
        
    attn_hook = model.cls_head_list[0].pooling.cross_attention.register_forward_hook(hook_attn)
    
    # -------------------------------------------------------------
    # Method A: 3D Grad-CAM & Method C: Cross-Attention
    # -------------------------------------------------------------
    bottleneck_var = bottleneck_act.detach().clone().requires_grad_(True)
    logit = model.cls_head_list[0](bottleneck_var)[0, 0]
    target_logit_val = logit.item()
    target_prob_val = torch.sigmoid(logit).item()
    
    logit.backward()
    grad_1 = bottleneck_var.grad.detach().clone()
    
    # Grad-CAM computation
    alpha_cam = grad_1.mean(dim=(2, 3, 4), keepdim=True) # (1, 320, 1, 1, 1)
    cam_7 = torch.relu((alpha_cam * bottleneck_act).sum(dim=1, keepdim=True)) # (1, 1, 7, 7, 7)
    cam_224 = F.interpolate(cam_7, size=(224, 224, 224), mode='trilinear', align_corners=False)
    results['gradcam'] = normalize_attribution(cam_224)
    
    # Cross-Attention computation
    raw_attn = attn_dict['weights'].squeeze(0) # (2, 343)
    avg_attn = raw_attn.mean(dim=0).reshape(1, 1, 7, 7, 7) # (1, 1, 7, 7, 7)
    attn_224 = F.interpolate(avg_attn, size=(224, 224, 224), mode='trilinear', align_corners=False)
    results['cross_attention'] = normalize_attribution(attn_224)
    attn_hook.remove()
    
    # -------------------------------------------------------------
    # Method B: 3D Grad-CAM++
    # -------------------------------------------------------------
    # Second-order gradients computation
    # Y = logit, A = bottleneck_var
    # alpha_c^(d,h,w) = (d^2 Y / dA^2) / (2 * d^2 Y / dA^2 + sum(A * d^3 Y / dA^3) + eps)
    # In CrossAttentionPooling, the attention output attended = LayerNorm(softmax(QK^T)*V)
    # The second derivative w.r.t. A is computed via autograd grad of grad_1
    bottleneck_var2 = bottleneck_act.detach().clone().requires_grad_(True)
    logit2 = model.cls_head_list[0](bottleneck_var2)[0, 0]
    grad_1_active = torch.autograd.grad(logit2, bottleneck_var2, create_graph=True)[0]
    
    # Second order: sum of grad_1 elements
    grad_2 = torch.autograd.grad(grad_1_active.sum(), bottleneck_var2, retain_graph=False)[0]
    
    # Grad-CAM++ alpha weighting
    denom = 2.0 * grad_2.pow(2) + 1e-7
    alpha_pp = torch.relu(grad_2) / denom
    alpha_pp = alpha_pp.mean(dim=(2, 3, 4), keepdim=True)
    cam_pp_7 = torch.relu((alpha_pp * bottleneck_act).sum(dim=1, keepdim=True))
    cam_pp_224 = F.interpolate(cam_pp_7, size=(224, 224, 224), mode='trilinear', align_corners=False)
    results['gradcam_plusplus'] = normalize_attribution(cam_pp_224)
    
    # -------------------------------------------------------------
    # Method D: Layer Integrated Gradients (Bottleneck IG, m=20)
    # -------------------------------------------------------------
    # Baseline: A' = 0
    m_steps = 20
    baseline_act = torch.zeros_like(bottleneck_act)
    accum_grad = torch.zeros_like(bottleneck_act)
    
    for step in range(1, m_steps + 1):
        interp_act = baseline_act + (float(step) / m_steps) * (bottleneck_act - baseline_act)
        interp_act = interp_act.detach().requires_grad_(True)
        step_logit = model.cls_head_list[0](interp_act)[0, 0]
        step_logit.backward()
        accum_grad += interp_act.grad
        
    avg_grad = accum_grad / m_steps
    ig_7 = torch.relu(((bottleneck_act - baseline_act) * avg_grad).sum(dim=1, keepdim=True))
    ig_224 = F.interpolate(ig_7, size=(224, 224, 224), mode='trilinear', align_corners=False)
    results['integrated_gradients'] = normalize_attribution(ig_224)
    
    return results, target_logit_val, target_prob_val

def run_pilot():
    print("=" * 70)
    print("PHASE 3: EXPLAINABILITY & ATTRIBUTION AUDIT — PILOT EXECUTION")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = load_p3_model(device)
    
    pilot_records = []
    case_timings = []
    
    for item in PILOT_CASES:
        cid = item["case_id"]
        cat = item["category"]
        print(f"\nProcessing Pilot Case: {cid} [{cat}]...")
        
        t0 = time.time()
        img_p = os.path.join(TOPANEU_DIR, "images", f"{cid}_0000.nii.gz")
        vess_p = os.path.join(TOPANEU_DIR, "vessel_masks", f"{cid}.nii.gz")
        loc_p = os.path.join(TOPANEU_DIR, "location_masks", f"{cid}.nii.gz")
        
        # Load and preprocess image
        nii_img = nib.load(img_p)
        raw_vol = nii_img.get_fdata().astype(np.float32)
        vol_tensor = torch.from_numpy(raw_vol).unsqueeze(0).unsqueeze(0).to(device)
        vol_224 = F.interpolate(vol_tensor, size=(224, 224, 224), mode='trilinear', align_corners=True)
        vol_norm = (vol_224 - vol_224.mean()) / vol_224.std().clamp(min=1e-8)
        
        # Load Ground Truth Masks
        gt_vess_data = nib.load(vess_p).get_fdata()
        gt_loc_data = nib.load(loc_p).get_fdata()
        vess_tensor = torch.from_numpy(gt_vess_data).unsqueeze(0).unsqueeze(0).float().to(device)
        loc_tensor = torch.from_numpy(gt_loc_data).unsqueeze(0).unsqueeze(0).float().to(device)
        
        gt_vess_224 = F.interpolate(vess_tensor, size=(224, 224, 224), mode='nearest').squeeze().cpu().numpy() > 0
        gt_loc_224 = F.interpolate(loc_tensor, size=(224, 224, 224), mode='nearest').squeeze().cpu().numpy() > 0
        
        torch.cuda.reset_peak_memory_stats()
        # Compute Attributions
        attrs, logit_val, prob_val = compute_attribution_methods(model, vol_norm, device)
        elapsed = time.time() - t0
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        case_timings.append(elapsed)
        
        # Compute basic overlap metrics
        metrics_case = {
            "case_id": cid,
            "category": cat,
            "presence_logit": logit_val,
            "presence_prob": prob_val,
            "elapsed_sec": round(elapsed, 3),
            "peak_vram_gb": round(peak_vram, 3)
        }
        
        for method_name, attr_map in attrs.items():
            attr_np = attr_map.cpu().numpy()
            total_mass = float(attr_np.sum())
            
            # Aneurysm overlap
            aneurysm_mass = float(attr_np[gt_loc_224].sum()) if gt_loc_224.any() else 0.0
            aneurysm_overlap = (aneurysm_mass / total_mass) if total_mass > 0 else 0.0
            
            # Vessel containment
            vessel_mass = float(attr_np[gt_vess_224].sum()) if gt_vess_224.any() else 0.0
            vessel_containment = (vessel_mass / total_mass) if total_mass > 0 else 0.0
            
            # Background mass (outside vessels and aneurysms)
            bg_mask = ~(gt_vess_224 | gt_loc_224)
            bg_mass = float(attr_np[bg_mask].sum()) if bg_mask.any() else 0.0
            bg_ratio = (bg_mass / total_mass) if total_mass > 0 else 0.0
            
            metrics_case[f"{method_name}_aneurysm_overlap"] = round(aneurysm_overlap, 5)
            metrics_case[f"{method_name}_vessel_containment"] = round(vessel_containment, 5)
            metrics_case[f"{method_name}_bg_ratio"] = round(bg_ratio, 5)
            
            # Verify numerical validity
            assert np.isfinite(attr_np).all(), f"NaN or Inf found in {method_name} for {cid}"
            assert attr_np.min() >= -1e-6 and attr_np.max() <= 1.0 + 1e-6, f"Invalid range in {method_name}"
            
        pilot_records.append(metrics_case)
        print(f"  Presence Logit: {logit_val:.4f} | Prob: {prob_val:.4f}")
        print(f"  Grad-CAM Aneurysm Overlap: {metrics_case['gradcam_aneurysm_overlap']:.4f} | Vessel Containment: {metrics_case['gradcam_vessel_containment']:.4f}")
        print(f"  Cross-Attention Aneurysm Overlap: {metrics_case['cross_attention_aneurysm_overlap']:.4f} | Vessel Containment: {metrics_case['cross_attention_vessel_containment']:.4f}")
        print(f"  Layer IG Aneurysm Overlap: {metrics_case['integrated_gradients_aneurysm_overlap']:.4f} | Vessel Containment: {metrics_case['integrated_gradients_vessel_containment']:.4f}")
        print(f"  Completed in {elapsed:.2f}s (Peak VRAM: {peak_vram:.2f} GB)")
        
    pilot_df = pd.DataFrame(pilot_records)
    csv_out = os.path.join(OUT_PILOT_DIR, "phase3_pilot_metrics.csv")
    pilot_df.to_csv(csv_out, index=False)
    print("\n" + "=" * 70)
    print(f"PILOT METRICS SAVED TO: {csv_out}")
    print(f"Average runtime per case: {np.mean(case_timings):.2f}s")
    print(f"Max peak VRAM observed: {max(r['peak_vram_gb'] for r in pilot_records):.2f} GB")
    print("ALL NUMERICAL CHECKS PASSED: Zero NaNs, valid normalization, finite gradients.")
    print("=" * 70)

if __name__ == "__main__":
    run_pilot()
