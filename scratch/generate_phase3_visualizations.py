import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import center_of_mass

# Configure module paths
sys.path.insert(0, r'D:\NLP_Project\scratch\dna_src\dynamic_network_architectures-0.3.1')
sys.path.insert(0, r'D:\NLP_Project\RSNA2025_Intracranial-Aneurysm-Detection\nnXNet')

from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet_two_seg_with_cls_modality import (
    ResEncoderUNet_two_seg_with_cls_modality
)

TOPANEU_DIR = r"D:\NLP_Project\topaneu_release"
STAGE2_CKPT = r"D:\NLP_Project\scratch\checkpoints\Dataset660_26classes_resize224_4661\onlyMirror01_lr4e3_100epochs_ps224\fold_0\checkpoint_final.pth"
OUT_VIS_DIR = r"D:\NLP_Project\scratch\phase3_visualizations"
os.makedirs(OUT_VIS_DIR, exist_ok=True)

REPRESENTATIVE_CASES = [
    {"case_id": "topaneu_center1_mr_056", "title": "True Positive — Exact Location Match (Right Supraclinoid ICA)"},
    {"case_id": "topaneu_center1_mr_148", "title": "True Positive — Prominent Saccular Aneurysm"},
    {"case_id": "topaneu_center1_mr_017", "title": "False Negative — Vessel-Aligned Parent (R ICA C7-PCom)"},
    {"case_id": "topaneu_center1_mr_028", "title": "False Negative — Saccular Aneurysm (Right ICA C7)"},
    {"case_id": "topaneu_center1_mr_018", "title": "False Positive — Candidate Spurious Parenchymal Attribution"},
    {"case_id": "topaneu_center1_mr_001", "title": "True Negative — Clean Control with Nonvascular Baseline"}
]

def load_p3_model(device):
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

def compute_visualizations_for_case(model, cid, device):
    img_p = os.path.join(TOPANEU_DIR, "images", f"{cid}_0000.nii.gz")
    vess_p = os.path.join(TOPANEU_DIR, "vessel_masks", f"{cid}.nii.gz")
    loc_p = os.path.join(TOPANEU_DIR, "location_masks", f"{cid}.nii.gz")
    
    raw_vol = nib.load(img_p).get_fdata().astype(np.float32)
    vol_tensor = torch.from_numpy(raw_vol).unsqueeze(0).unsqueeze(0).to(device)
    vol_224 = F.interpolate(vol_tensor, size=(224, 224, 224), mode='trilinear', align_corners=True)
    vol_norm = (vol_224 - vol_224.mean()) / vol_224.std().clamp(min=1e-8)
    
    gt_vess_data = nib.load(vess_p).get_fdata()
    gt_loc_data = nib.load(loc_p).get_fdata()
    vess_t = torch.from_numpy(gt_vess_data).unsqueeze(0).unsqueeze(0).float().to(device)
    loc_t = torch.from_numpy(gt_loc_data).unsqueeze(0).unsqueeze(0).float().to(device)
    
    gt_vess_224 = F.interpolate(vess_t, size=(224, 224, 224), mode='nearest').squeeze().cpu().numpy() > 0
    gt_loc_224 = F.interpolate(loc_t, size=(224, 224, 224), mode='nearest').squeeze().cpu().numpy() > 0
    
    # Forward encoder
    with torch.no_grad():
        conv_enc_outputs = [model.conv_encoder_blocks[0](vol_norm)]
        for i in range(1, len(model.conv_encoder_blocks)):
            conv_enc_outputs.append(model.conv_encoder_blocks[i](conv_enc_outputs[-1]))
        bottleneck_act = conv_enc_outputs[-1]
        
    attn_dict = {}
    def hook_attn(module, inp, out):
        attn_dict['weights'] = out[1]
    h = model.cls_head_list[0].pooling.cross_attention.register_forward_hook(hook_attn)
    
    bottleneck_var = bottleneck_act.detach().clone().requires_grad_(True)
    logit = model.cls_head_list[0](bottleneck_var)[0, 0]
    prob_val = torch.sigmoid(logit).item()
    logit_val = logit.item()
    logit.backward()
    grad_1 = bottleneck_var.grad.detach().clone()
    
    # 1. Grad-CAM
    alpha_cam = grad_1.mean(dim=(2, 3, 4), keepdim=True)
    cam_7 = torch.relu((alpha_cam * bottleneck_act).sum(dim=1, keepdim=True))
    cam_224 = normalize_attribution(F.interpolate(cam_7, size=(224, 224, 224), mode='trilinear', align_corners=False)).cpu().numpy()
    
    # 2. Cross-Attention
    raw_attn = attn_dict['weights'].squeeze(0)
    avg_attn = raw_attn.mean(dim=0).reshape(1, 1, 7, 7, 7)
    attn_224 = normalize_attribution(F.interpolate(avg_attn, size=(224, 224, 224), mode='trilinear', align_corners=False)).cpu().numpy()
    h.remove()
    
    # 3. Grad-CAM++
    bottleneck_var2 = bottleneck_act.detach().clone().requires_grad_(True)
    logit2 = model.cls_head_list[0](bottleneck_var2)[0, 0]
    grad_1_active = torch.autograd.grad(logit2, bottleneck_var2, create_graph=True)[0]
    grad_2 = torch.autograd.grad(grad_1_active.sum(), bottleneck_var2, retain_graph=False)[0]
    denom = 2.0 * grad_2.pow(2) + 1e-7
    alpha_pp = torch.relu(grad_2) / denom
    alpha_pp = alpha_pp.mean(dim=(2, 3, 4), keepdim=True)
    cam_pp_7 = torch.relu((alpha_pp * bottleneck_act).sum(dim=1, keepdim=True))
    cam_pp_224 = normalize_attribution(F.interpolate(cam_pp_7, size=(224, 224, 224), mode='trilinear', align_corners=False)).cpu().numpy()
    
    # 4. Layer IG (m=20)
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
    ig_224 = normalize_attribution(F.interpolate(ig_7, size=(224, 224, 224), mode='trilinear', align_corners=False)).cpu().numpy()
    
    raw_vol_np = vol_norm.squeeze().cpu().numpy()
    return raw_vol_np, gt_vess_224, gt_loc_224, cam_224, cam_pp_224, attn_224, ig_224, logit_val, prob_val

def render_publication_figure(cid, title, raw_vol, gt_vess, gt_loc, cam, cam_pp, attn, ig, logit, prob):
    # Determine focal slice along axial plane (Z dimension)
    if gt_loc.any():
        c_z, c_y, c_x = center_of_mass(gt_loc)
        focal_z = int(round(c_z))
    elif gt_vess.any():
        c_z, c_y, c_x = center_of_mass(gt_vess)
        focal_z = int(round(c_z))
    else:
        focal_z = 112
        
    focal_z = np.clip(focal_z, 10, 214)
    
    # Extract 2D slices
    raw_slice = raw_vol[focal_z, :, :]
    gt_vess_slice = gt_vess[focal_z, :, :]
    gt_loc_slice = gt_loc[focal_z, :, :]
    cam_slice = cam[focal_z, :, :]
    cam_pp_slice = cam_pp[focal_z, :, :]
    attn_slice = attn[focal_z, :, :]
    ig_slice = ig[focal_z, :, :]
    
    # Create 2x3 Grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='black')
    fig.suptitle(
        f"Phase 3 XAI Diagnostic: {cid}\n{title} | Focal Slice Z={focal_z} | P(presence)={prob:.4f} (logit={logit:.2f})",
        fontsize=16, color='white', fontweight='bold', y=0.98
    )
    
    # Colormaps
    heat_cmap = plt.cm.turbo
    
    def plot_slice(ax, base, overlay=None, ov_cmap=None, title_str=""):
        ax.set_facecolor('black')
        ax.imshow(base, cmap='gray', origin='lower')
        if overlay is not None:
            # Mask low values for transparency
            masked_ov = np.ma.masked_where(overlay < 0.1, overlay)
            ax.imshow(masked_ov, cmap=ov_cmap, alpha=0.55, origin='lower')
        ax.set_title(title_str, color='white', fontsize=12, fontweight='bold', pad=8)
        ax.axis('off')
        
    # Panel 1: Raw Image
    plot_slice(axes[0, 0], raw_slice, title_str="1. Raw MRA Axial Slice")
    
    # Panel 2: Ground Truth Anatomy Overlay
    axes[0, 1].set_facecolor('black')
    axes[0, 1].imshow(raw_slice, cmap='gray', origin='lower')
    if gt_vess_slice.any():
        vess_mask = np.ma.masked_where(~gt_vess_slice, gt_vess_slice)
        axes[0, 1].imshow(vess_mask, cmap='Greens', alpha=0.5, origin='lower')
    if gt_loc_slice.any():
        loc_mask = np.ma.masked_where(~gt_loc_slice, gt_loc_slice)
        axes[0, 1].imshow(loc_mask, cmap='Reds', alpha=0.85, origin='lower')
    axes[0, 1].set_title("2. Ground Truth (Green=Vessel, Red=Aneurysm)", color='white', fontsize=12, fontweight='bold', pad=8)
    axes[0, 1].axis('off')
    
    # Panel 3: 3D Grad-CAM Overlay
    plot_slice(axes[0, 2], raw_slice, cam_slice, heat_cmap, "3. 3D Grad-CAM (Bottleneck)")
    
    # Panel 4: 3D Grad-CAM++ Overlay
    plot_slice(axes[1, 0], raw_slice, cam_pp_slice, heat_cmap, "4. 3D Grad-CAM++ (Second Order)")
    
    # Panel 5: Cross-Attention Spatial Map
    plot_slice(axes[1, 1], raw_slice, attn_slice, heat_cmap, "5. Cross-Attention Query Map")
    
    # Panel 6: Layer Integrated Gradients
    plot_slice(axes[1, 2], raw_slice, ig_slice, heat_cmap, "6. Layer Integrated Gradients (m=20)")
    
    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.94])
    out_path = os.path.join(OUT_VIS_DIR, f"{cid}_phase3_xai_diagnostic.png")
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"Rendered diagnostic figure saved to: {out_path}", flush=True)

def main():
    print("=" * 70)
    print("GENERATING PHASE 3 REPRESENTATIVE PUBLICATION FIGURES")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_p3_model(device)
    
    for item in REPRESENTATIVE_CASES:
        cid = item["case_id"]
        title = item["title"]
        print(f"\nGenerating figure for {cid} ({title})...", flush=True)
        raw_vol, gt_vess, gt_loc, cam, cam_pp, attn, ig, logit, prob = compute_visualizations_for_case(model, cid, device)
        render_publication_figure(cid, title, raw_vol, gt_vess, gt_loc, cam, cam_pp, attn, ig, logit, prob)
        
    print("\n" + "=" * 70)
    print("ALL PUBLICATION DIAGNOSTIC FIGURES GENERATED SUCCESSFULLY.")
    print("=" * 70)

if __name__ == "__main__":
    main()
