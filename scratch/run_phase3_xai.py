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
from scipy.ndimage import center_of_mass

# Configure module paths
sys.path.insert(0, r'D:\NLP_Project\scratch\dna_src\dynamic_network_architectures-0.3.1')
sys.path.insert(0, r'D:\NLP_Project\RSNA2025_Intracranial-Aneurysm-Detection\nnXNet')

from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet_two_seg_with_cls_modality import (
    ResEncoderUNet_two_seg_with_cls_modality
)

TOPANEU_DIR = r"D:\NLP_Project\topaneu_release"
STAGE2_CKPT = r"D:\NLP_Project\scratch\checkpoints\Dataset660_26classes_resize224_4661\onlyMirror01_lr4e3_100epochs_ps224\fold_0\checkpoint_final.pth"
PRED_DIR = r"D:\NLP_Project\scratch\phase2_segmentation_predictions"
BASELINE_CSV = r"D:\NLP_Project\p3_pretrained_topaneu_predictions.csv"
MAPPING_CSV = r"D:\NLP_Project\PHASE2_LABEL_MAPPING.csv"
OUT_METRICS_CSV = r"D:\NLP_Project\PHASE3_XAI_METRICS.csv"
OUT_COMPARISON_CSV = r"D:\NLP_Project\PHASE3_XAI_METHOD_COMPARISON.csv"
VIS_DIR = r"D:\NLP_Project\scratch\phase3_visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

# 40-Case Evaluation Cohort from Phase 2
COHORT_CASES = [
    f.replace(".npz", "") for f in os.listdir(PRED_DIR) if f.endswith(".npz")
]

P3_LOC_NAMES = [
    "Other Posterior Circulation", "Basilar Tip", "Right Posterior Communicating Artery",
    "Left Posterior Communicating Artery", "Right Infraclinoid Internal Carotid Artery",
    "Left Infraclinoid Internal Carotid Artery", "Right Supraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery", "Right Middle Cerebral Artery",
    "Left Middle Cerebral Artery", "Right Anterior Cerebral Artery",
    "Left Anterior Cerebral Artery", "Anterior Communicating Artery"
]

def load_p3_model(device):
    print("Loading ResEncoderUNet_two_seg_with_cls_modality for Phase 3...")
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
    print(f"Pretrained weights loaded strictly from {STAGE2_CKPT}.")
    return model

def normalize_attribution(attr_tensor):
    attr = attr_tensor.squeeze().detach()
    a_min = attr.min()
    a_max = attr.max()
    if a_max - a_min > 1e-8:
        norm_attr = (attr - a_min) / (a_max - a_min)
    else:
        norm_attr = torch.zeros_like(attr)
    return norm_attr

def compute_all_attributions(model, vol_norm):
    """
    Computes 3D Grad-CAM, 3D Grad-CAM++, Cross-Attention, and Layer IG
    """
    results = {}
    
    # Forward encoder under torch.no_grad()
    with torch.no_grad():
        conv_enc_outputs = [model.conv_encoder_blocks[0](vol_norm)]
        for i in range(1, len(model.conv_encoder_blocks)):
            conv_enc_outputs.append(model.conv_encoder_blocks[i](conv_enc_outputs[-1]))
        bottleneck_act = conv_enc_outputs[-1] # (1, 320, 7, 7, 7)
        
    # Hook cross attention
    attn_dict = {}
    def hook_attn(module, inp, out):
        attn_dict['weights'] = out[1] # (1, 2, 343)
    attn_hook = model.cls_head_list[0].pooling.cross_attention.register_forward_hook(hook_attn)
    
    # Forward through presence head
    bottleneck_var = bottleneck_act.detach().clone().requires_grad_(True)
    logit = model.cls_head_list[0](bottleneck_var)[0, 0]
    target_logit_val = logit.item()
    target_prob_val = torch.sigmoid(logit).item()
    
    logit.backward()
    grad_1 = bottleneck_var.grad.detach().clone()
    
    # 1. 3D Grad-CAM
    alpha_cam = grad_1.mean(dim=(2, 3, 4), keepdim=True)
    cam_7 = torch.relu((alpha_cam * bottleneck_act).sum(dim=1, keepdim=True))
    cam_224 = F.interpolate(cam_7, size=(224, 224, 224), mode='trilinear', align_corners=False)
    results['gradcam'] = normalize_attribution(cam_224)
    
    # 2. Cross-Attention
    raw_attn = attn_dict['weights'].squeeze(0) # (2, 343)
    avg_attn = raw_attn.mean(dim=0).reshape(1, 1, 7, 7, 7)
    attn_224 = F.interpolate(avg_attn, size=(224, 224, 224), mode='trilinear', align_corners=False)
    results['cross_attention'] = normalize_attribution(attn_224)
    attn_hook.remove()
    
    # 3. 3D Grad-CAM++
    bottleneck_var2 = bottleneck_act.detach().clone().requires_grad_(True)
    logit2 = model.cls_head_list[0](bottleneck_var2)[0, 0]
    grad_1_active = torch.autograd.grad(logit2, bottleneck_var2, create_graph=True)[0]
    grad_2 = torch.autograd.grad(grad_1_active.sum(), bottleneck_var2, retain_graph=False)[0]
    
    denom = 2.0 * grad_2.pow(2) + 1e-7
    alpha_pp = torch.relu(grad_2) / denom
    alpha_pp = alpha_pp.mean(dim=(2, 3, 4), keepdim=True)
    cam_pp_7 = torch.relu((alpha_pp * bottleneck_act).sum(dim=1, keepdim=True))
    cam_pp_224 = F.interpolate(cam_pp_7, size=(224, 224, 224), mode='trilinear', align_corners=False)
    results['gradcam_plusplus'] = normalize_attribution(cam_pp_224)
    
    # 4. Layer Integrated Gradients (m=20)
    m_steps = 20
    baseline_act = torch.zeros_like(bottleneck_act)
    accum_grad = torch.zeros_like(bottleneck_act)
    for step in range(1, m_steps + 1):
        interp_act = baseline_act + (float(step) / m_steps) * (bottleneck_act - baseline_act)
        interp_act = interp_act.detach().requires_grad_(True)
        s_logit = model.cls_head_list[0](interp_act)[0, 0]
        s_logit.backward()
        accum_grad += interp_act.grad
    avg_grad = accum_grad / m_steps
    ig_7 = torch.relu(((bottleneck_act - baseline_act) * avg_grad).sum(dim=1, keepdim=True))
    ig_224 = F.interpolate(ig_7, size=(224, 224, 224), mode='trilinear', align_corners=False)
    results['integrated_gradients'] = normalize_attribution(ig_224)
    
    return results, target_logit_val, target_prob_val

def compute_centroid(mask):
    """Compute 3D center of mass coordinates in voxel space."""
    if not mask.any():
        return None
    c = center_of_mass(mask)
    return np.array(c)

def main():
    print("=" * 75)
    print("PHASE 3: FULL COHORT XAI EXECUTION (40-CASE SEGMENTATION-GROUNDED COHORT)")
    print("=" * 75)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device} ({torch.cuda.get_device_name(0)})")
    
    model = load_p3_model(device)
    base_df = pd.read_csv(BASELINE_CSV).set_index("case_id")
    map_df = pd.read_csv(MAPPING_CSV)
    loc_rules = map_df[map_df["category"] == "Location (Aneurysm)"].set_index("topaneu_id")
    vess_rules = map_df[map_df["category"] == "Vessel (Anatomy)"].set_index("topaneu_id")
    
    all_metrics = []
    
    # Iterate through all 40 cohort cases
    for idx, cid in enumerate(sorted(COHORT_CASES), 1):
        t0 = time.time()
        print(f"[{idx}/{len(COHORT_CASES)}] Processing Case: {cid}...", flush=True)
        
        # Load Raw Volume
        img_p = os.path.join(TOPANEU_DIR, "images", f"{cid}_0000.nii.gz")
        vess_p = os.path.join(TOPANEU_DIR, "vessel_masks", f"{cid}.nii.gz")
        loc_p = os.path.join(TOPANEU_DIR, "location_masks", f"{cid}.nii.gz")
        
        nii_img = nib.load(img_p)
        raw_vol = nii_img.get_fdata().astype(np.float32)
        vol_tensor = torch.from_numpy(raw_vol).unsqueeze(0).unsqueeze(0).to(device)
        vol_224 = F.interpolate(vol_tensor, size=(224, 224, 224), mode='trilinear', align_corners=True)
        vol_norm = (vol_224 - vol_224.mean()) / vol_224.std().clamp(min=1e-8)
        
        # Load GT Masks (nearest neighbor resampled to 224^3)
        gt_vess_data = nib.load(vess_p).get_fdata()
        gt_loc_data = nib.load(loc_p).get_fdata()
        vess_t = torch.from_numpy(gt_vess_data).unsqueeze(0).unsqueeze(0).float().to(device)
        loc_t = torch.from_numpy(gt_loc_data).unsqueeze(0).unsqueeze(0).float().to(device)
        
        gt_vess_224 = F.interpolate(vess_t, size=(224, 224, 224), mode='nearest').squeeze().cpu().numpy().astype(np.uint8)
        gt_loc_224 = F.interpolate(loc_t, size=(224, 224, 224), mode='nearest').squeeze().cpu().numpy().astype(np.uint8)
        
        gt_aneurysm_bin = (gt_loc_224 > 0)
        gt_vessel_bin = (gt_vess_224 > 0)
        
        # Load Phase-2 Predicted Segmentations
        p2_npz = np.load(os.path.join(PRED_DIR, f"{cid}.npz"))
        seg1_raw = p2_npz["seg1_vessel"]
        pred_seg1_vessel = ((seg1_raw >= 1) & (seg1_raw <= 13))
        pred_seg1_aneurysm = (seg1_raw == 14)
        
        # Metadata from baseline
        b_row = base_df.loc[cid]
        gt_presence = int(b_row["ground_truth_presence"])
        pred_prob = float(b_row["predicted_presence_probability"])
        pred_loc = str(b_row["predicted_location"])
        
        # Stratification label
        if gt_presence == 1:
            if pred_prob >= 0.5:
                strat = "TP"
            else:
                strat = "FN"
        else:
            if pred_prob >= 0.5:
                strat = "FP"
            else:
                strat = "TN"
                
        # Compute Attributions
        attrs, logit_val, prob_val = compute_all_attributions(model, vol_norm)
        gt_aneurysm_centroid = compute_centroid(gt_aneurysm_bin)
        
        # Evaluate each method
        for method_name, attr_map in attrs.items():
            attr_np = attr_map.cpu().numpy()
            total_mass = float(attr_np.sum())
            
            # 1. Aneurysm attribution overlap
            aneurysm_mass = float(attr_np[gt_aneurysm_bin].sum()) if gt_aneurysm_bin.any() else 0.0
            aneurysm_overlap = (aneurysm_mass / total_mass) if total_mass > 0 else 0.0
            
            # Top 5% attribution mass overlap with aneurysm
            top5_thresh = np.percentile(attr_np, 95)
            top5_mask = (attr_np >= top5_thresh)
            top5_aneurysm_overlap = float((top5_mask & gt_aneurysm_bin).sum()) / float(gt_aneurysm_bin.sum()) if gt_aneurysm_bin.any() else 0.0
            
            # 2. Vessel containment (GT vessel tree)
            vessel_mass = float(attr_np[gt_vessel_bin].sum()) if gt_vessel_bin.any() else 0.0
            vessel_containment = (vessel_mass / total_mass) if total_mass > 0 else 0.0
            
            # 3. Background / parenchymal attribution
            bg_mask = ~(gt_vessel_bin | gt_aneurysm_bin)
            bg_mass = float(attr_np[bg_mask].sum()) if bg_mask.any() else 0.0
            bg_attribution = (bg_mass / total_mass) if total_mass > 0 else 0.0
            
            # 4. Centroid Distance (in mm, 1mm isotropic)
            attr_centroid = compute_centroid(top5_mask)
            if gt_aneurysm_centroid is not None and attr_centroid is not None:
                centroid_dist_mm = float(np.linalg.norm(attr_centroid - gt_aneurysm_centroid))
            else:
                centroid_dist_mm = np.nan
                
            # 5. Overlap with P3's own predicted segmentations
            pred_vessel_mass = float(attr_np[pred_seg1_vessel].sum()) if pred_seg1_vessel.any() else 0.0
            pred_vessel_overlap = (pred_vessel_mass / total_mass) if total_mass > 0 else 0.0
            
            pred_aneurysm_mass = float(attr_np[pred_seg1_aneurysm].sum()) if pred_seg1_aneurysm.any() else 0.0
            pred_aneurysm_overlap = (pred_aneurysm_mass / total_mass) if total_mass > 0 else 0.0
            
            # 6. Attention agreement (Pearson correlation with cross_attention)
            attn_np = attrs['cross_attention'].cpu().numpy()
            corr = float(np.corrcoef(attr_np.flatten(), attn_np.flatten())[0, 1]) if method_name != 'cross_attention' else 1.0
            
            all_metrics.append({
                "case_id": cid,
                "cohort_stratum": strat,
                "ground_truth_presence": gt_presence,
                "predicted_presence_probability": pred_prob,
                "presence_logit": logit_val,
                "predicted_location": pred_loc,
                "xai_method": method_name,
                "aneurysm_attribution_overlap": round(aneurysm_overlap, 6),
                "top5_aneurysm_overlap": round(top5_aneurysm_overlap, 6),
                "vessel_containment": round(vessel_containment, 6),
                "background_attribution": round(bg_attribution, 6),
                "attribution_centroid_distance_mm": round(centroid_dist_mm, 2) if np.isfinite(centroid_dist_mm) else np.nan,
                "predicted_vessel_overlap": round(pred_vessel_overlap, 6),
                "predicted_aneurysm_overlap": round(pred_aneurysm_overlap, 6),
                "attention_agreement_corr": round(corr, 4),
                "runtime_seconds": round(time.time() - t0, 3)
            })
            
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(OUT_METRICS_CSV, index=False)
    print(f"\nSaved all case-level XAI metrics to: {OUT_METRICS_CSV}")
    
    # Method-level summary comparison
    print("\n" + "=" * 75)
    print("METHOD-LEVEL SUMMARY COMPARISON (AVERAGED OVER 40 CASES)")
    print("=" * 75)
    methods = ["gradcam", "gradcam_plusplus", "cross_attention", "integrated_gradients"]
    comp_rows = []
    
    for m in methods:
        sub = df_metrics[df_metrics["xai_method"] == m]
        sub_pos = sub[sub["ground_truth_presence"] == 1]
        
        row = {
            "xai_method": m,
            "mean_aneurysm_overlap": round(sub_pos["aneurysm_attribution_overlap"].mean(), 6),
            "median_aneurysm_overlap": round(sub_pos["aneurysm_attribution_overlap"].median(), 6),
            "mean_top5_aneurysm_overlap": round(sub_pos["top5_aneurysm_overlap"].mean(), 6),
            "mean_vessel_containment": round(sub["vessel_containment"].mean(), 6),
            "mean_background_attribution": round(sub["background_attribution"].mean(), 6),
            "mean_centroid_distance_mm": round(sub_pos["attribution_centroid_distance_mm"].dropna().mean(), 2),
            "median_centroid_distance_mm": round(sub_pos["attribution_centroid_distance_mm"].dropna().median(), 2),
            "mean_predicted_vessel_overlap": round(sub["predicted_vessel_overlap"].mean(), 6),
            "mean_predicted_aneurysm_overlap": round(sub_pos["predicted_aneurysm_overlap"].mean(), 6),
            "mean_attention_agreement_corr": round(sub["attention_agreement_corr"].mean(), 4)
        }
        comp_rows.append(row)
        print(f"Method: {m.upper()}")
        print(f"  Aneurysm Overlap: {row['mean_aneurysm_overlap']:.6f} (Top-5% overlap: {row['mean_top5_aneurysm_overlap']:.6f})")
        print(f"  Vessel Containment: {row['mean_vessel_containment']:.4f} | Background Attr: {row['mean_background_attribution']:.4f}")
        print(f"  Centroid Dist: {row['mean_centroid_distance_mm']} mm (Median: {row['median_centroid_distance_mm']} mm)")
        print(f"  Attn Correlation: {row['mean_attention_agreement_corr']:.4f}")
        
    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(OUT_COMPARISON_CSV, index=False)
    print(f"Saved method comparison table to: {OUT_COMPARISON_CSV}")
    print("=" * 75)

if __name__ == "__main__":
    main()
