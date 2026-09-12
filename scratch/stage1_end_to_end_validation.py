import os
import sys
import json
import time
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure module search paths
sys.path.insert(0, r'D:\NLP_Project\scratch\dna_src\dynamic_network_architectures-0.3.1')
sys.path.insert(0, r'D:\NLP_Project\RSNA2025_Intracranial-Aneurysm-Detection\nnXNet')

import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnxnet.inference.predict_from_raw_data_2D_orthogonal_planes_fast import nnXNetPredictor
from nnxnet.imageio.simpleitk_reader_writer import SimpleITKIO
from nnxnet.utilities.plans_handling.plans_handler import PlansManager
from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet_two_seg_with_cls_modality import ResEncoderUNet_two_seg_with_cls_modality

# File Paths
TOPANEU_DIR = r"D:\NLP_Project\topaneu_release"
STAGE1_MODEL_DIR = r"D:\NLP_Project\scratch\checkpoints\Dataset180_2D_vessel_box_seg_stable\nnUNetTrainer__nnUNetPlans__2d"
STAGE1_CKPT = os.path.join(STAGE1_MODEL_DIR, "fold_0", "checkpoint_final.pth")
STAGE1_PLANS = os.path.join(STAGE1_MODEL_DIR, "plans.json")

STAGE2_CKPT = r"D:\NLP_Project\scratch\checkpoints\Dataset660_26classes_resize224_4661\onlyMirror01_lr4e3_100epochs_ps224\fold_0\checkpoint_final.pth"

OUTPUT_CSV = r"D:\NLP_Project\stage1_topaneu_predictions.csv"
VIS_DIR = r"D:\NLP_Project\scratch\stage1_roi_visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

BASELINE_CSV = r"D:\NLP_Project\p3_pretrained_topaneu_predictions.csv"

# Cases to validate
TARGET_CASES = [
    {"case_id": "topaneu_center1_mr_017", "is_positive": True},
    {"case_id": "topaneu_center1_mr_024", "is_positive": True},
    {"case_id": "topaneu_center1_mr_028", "is_positive": True},
    {"case_id": "topaneu_center1_mr_001", "is_positive": False},
]

P3_LOC_NAMES = [
    "Left Infraclinoid ICA",
    "Right Infraclinoid ICA",
    "Left Supraclinoid ICA",
    "Right Supraclinoid ICA",
    "Left MCA",
    "Right MCA",
    "Anterior Communicating Artery",
    "Left ACA",
    "Right ACA",
    "Left Pcom",
    "Right Pcom",
    "Basilar Tip",
    "Other Posterior Circulation"
]
MODALITY_NAMES = ["CTA", "MRA", "MRI T2", "MRI T1post"]


def load_stage1_predictor(device):
    print("Loading Stage-1 plans and constructing PlainConvUNet...")
    with open(STAGE1_PLANS, 'r') as f:
        plans = json.load(f)

    cfg = plans['configurations']['2d']
    arch_kwargs = cfg['architecture']['arch_kwargs']

    network = PlainConvUNet(
        input_channels=1,
        n_stages=arch_kwargs['n_stages'],
        features_per_stage=arch_kwargs['features_per_stage'],
        conv_op=nn.Conv2d,
        kernel_sizes=arch_kwargs['kernel_sizes'],
        strides=arch_kwargs['strides'],
        n_conv_per_stage=arch_kwargs['n_conv_per_stage'],
        num_classes=2,
        n_conv_per_stage_decoder=arch_kwargs['n_conv_per_stage_decoder'],
        conv_bias=arch_kwargs['conv_bias'],
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs=arch_kwargs['norm_op_kwargs'],
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=arch_kwargs['nonlin_kwargs'],
        deep_supervision=False
    )

    print(f"Strict loading Stage-1 checkpoint from: {STAGE1_CKPT}...")
    ckpt = torch.load(STAGE1_CKPT, map_location='cpu', weights_only=False)
    load_res = network.load_state_dict(ckpt['network_weights'], strict=True)
    print("Stage-1 strict load result:", load_res)

    predictor = nnXNetPredictor(
        tile_step_size=0.5,
        use_mirroring=False,
        use_gaussian=True,
        perform_everything_on_device=(device.type == 'cuda'),
        device=device,
        allow_tqdm=False
    )
    plans_manager = PlansManager(plans)
    predictor.plans_manager = plans_manager
    predictor.configuration_manager = plans_manager.get_configuration('2d')
    predictor.list_of_parameters = [ckpt['network_weights']]
    predictor.network = network
    predictor.initialize_network_and_gaussian()
    return predictor


def load_stage2_model(device):
    print("Constructing Stage-2 ResEncoderUNet_two_seg_with_cls_modality...")
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
        nonlin_kwargs={"inplace": True}, deep_supervision=True
    ).to(device)

    print(f"Strict loading Stage-2 checkpoint from: {STAGE2_CKPT}...")
    ckpt = torch.load(STAGE2_CKPT, map_location='cpu', weights_only=False)
    load_res = model.load_state_dict(ckpt['network_weights'], strict=True)
    print("Stage-2 strict load result:", load_res)
    model.eval()
    return model


def run_stage1_instrumented(predictor, input_image_np, original_spacing, target_spacing, max_batch_size=16):
    """
    Instrumented version of predict_from_multi_axial_slices to record detailed per-slice candidate,
    predicted vessel mask, and per-axis bounding boxes.
    """
    from nnxnet.preprocessing.resampling.resample_torch import resample_torch_simple
    from acvl_utils.cropping_and_padding.padding import pad_nd_image

    c, z_dim, y_dim, x_dim = input_image_np.shape

    all_patches = []
    all_slicers_info = []

    candidates_dict = {}
    slice_images_dict = {}

    for axis_name in ['Z', 'Y', 'X']:
        slices = input_image_np[0, :, :, :]

        if axis_name == 'Z':
            spacings = np.array([original_spacing[1], original_spacing[0]])
        elif axis_name == 'Y':
            slices = np.transpose(slices, (1, 0, 2))
            spacings = np.array([original_spacing[2], original_spacing[0]])
        else:  # axis_name == 'X'
            slices = np.transpose(slices, (2, 0, 1))
            spacings = np.array([original_spacing[2], original_spacing[1]])

        candidates = [slices.shape[0] // 2, slices.shape[0] // 4, slices.shape[0] * 3 // 4]
        non_empty_indices = [idx for idx in candidates if np.any(slices[idx])]
        candidates_dict[axis_name] = candidates

        for idx in non_empty_indices:
            raw_slice = slices[idx, :, :]
            slice_images_dict[(axis_name, idx)] = raw_slice
            image_slice_np = raw_slice[None][None]

            # 1. Normalization
            image_slice_np = (image_slice_np - image_slice_np.mean()) / (np.clip(image_slice_np.std(), 1e-8, None))

            # 2. Resampling
            current_spacing = np.concatenate([np.array([1]), spacings])
            dst_shape = [1, int(round(image_slice_np.shape[2] * current_spacing[1] / target_spacing[1])),
                        int(round(image_slice_np.shape[3] * current_spacing[2] / target_spacing[2]))]

            image_resized = resample_torch_simple(
                torch.from_numpy(image_slice_np).float().to(predictor.device),
                dst_shape,
                is_seg=False,
                device=predictor.device,
                mode='trilinear'
            )

            # 3. Padding and sliding window
            padded_image, slicer_revert_padding = pad_nd_image(
                image_resized, predictor.configuration_manager.patch_size, 'constant', {'value': 0}, True, None
            )
            slicers = predictor._internal_get_sliding_window_slicers(
                padded_image.shape[1:], predictor.configuration_manager.patch_size, predictor.tile_step_size
            )

            for i, sl in enumerate(slicers):
                patch = padded_image[sl]
                all_patches.append(patch)
                all_slicers_info.append({
                    'axis': axis_name,
                    'idx': idx,
                    'slicer': sl,
                    'padded_shape': padded_image.shape,
                    'original_shape': image_slice_np.shape,
                    'revert_padding': slicer_revert_padding
                })

    if not all_patches:
        # Fallback triggered immediately
        return (0, z_dim, 0, y_dim, 0, x_dim), True, candidates_dict, {}, []

    # Batch inference
    all_predictions = []
    for i in range(0, len(all_patches), max_batch_size):
        batch_patches = all_patches[i:i + max_batch_size]
        batch_tensor = torch.stack(batch_patches, dim=0).to(predictor.device)

        with torch.autocast(predictor.device.type, enabled=predictor.device.type == 'cuda'):
            with torch.no_grad():
                batch_pred = predictor.network(batch_tensor)
                all_predictions.append(batch_pred)

    full_batch_predictions = torch.cat(all_predictions, dim=0) if all_predictions else torch.tensor([])

    # Accumulate results
    aggregated_results = {}
    for info in all_slicers_info:
        key = (info['axis'], info['idx'])
        if key not in aggregated_results:
            num_classes = 2
            shape = info['padded_shape']
            aggregated_results[key] = {
                'logits': torch.zeros((num_classes, *shape[1:]), dtype=torch.half, device=predictor.device),
                'n_predictions': torch.zeros(shape[1:], dtype=torch.half, device=predictor.device),
                'info': info
            }

    for i, prediction in enumerate(full_batch_predictions):
        info = all_slicers_info[i]
        key = (info['axis'], info['idx'])
        if predictor.use_gaussian:
            prediction = prediction * predictor.gaussian
            weight = predictor.gaussian
        else:
            weight = 1.0

        sl = info['slicer']
        aggregated_results[key]['logits'][sl] += prediction
        aggregated_results[key]['n_predictions'][sl[1:]] += weight

    predictions = []
    predicted_masks_dict = {}

    for key, result in aggregated_results.items():
        info = result['info']
        logits = result['logits']
        n_predictions = result['n_predictions']

        logits.div_(n_predictions.unsqueeze(0))
        logits = logits[(slice(None), *info['revert_padding'][1:])]

        logits_resampled = resample_torch_simple(
            logits,
            info['original_shape'][1:],
            is_seg=False,
            device=predictor.device,
            mode='trilinear'
        )

        pred_array = logits_resampled.cpu().argmax(0).numpy().squeeze()
        predicted_masks_dict[key] = pred_array

        if np.any(pred_array > 0):
            predictions.append((info['axis'], 'center' if info['idx'] == input_image_np.shape[1] // 2 else 'sub', pred_array, info['idx']))

    if not predictions:
        return (0, z_dim, 0, y_dim, 0, x_dim), True, candidates_dict, predicted_masks_dict, []

    # Vectorized bbox collection & mean fusion
    final_z_min, final_z_max = [], []
    final_y_min, final_y_max = [], []
    final_x_min, final_x_max = [], []
    per_slice_bboxes = []

    for axis, _, pred_slice, idx in predictions:
        if axis == 'Z':
            z_min, z_max = idx, idx + 1
            y_coords, x_coords = np.where(pred_slice > 0)
            if len(y_coords) > 0:
                y_min, y_max = int(y_coords.min()), int(y_coords.max() + 1)
                x_min, x_max = int(x_coords.min()), int(x_coords.max() + 1)
                final_y_min.append(y_min); final_y_max.append(y_max)
                final_x_min.append(x_min); final_x_max.append(x_max)
                per_slice_bboxes.append({"axis": "Z", "idx": idx, "box": (z_min, z_max, y_min, y_max, x_min, x_max)})
        elif axis == 'Y':
            y_min, y_max = idx, idx + 1
            z_coords, x_coords = np.where(pred_slice > 0)
            if len(z_coords) > 0:
                z_min, z_max = int(z_coords.min()), int(z_coords.max() + 1)
                x_min, x_max = int(x_coords.min()), int(x_coords.max() + 1)
                final_z_min.append(z_min); final_z_max.append(z_max)
                final_x_min.append(x_min); final_x_max.append(x_max)
                per_slice_bboxes.append({"axis": "Y", "idx": idx, "box": (z_min, z_max, y_min, y_max, x_min, x_max)})
        elif axis == 'X':
            x_min, x_max = idx, idx + 1
            z_coords, y_coords = np.where(pred_slice > 0)
            if len(z_coords) > 0:
                z_min, z_max = int(z_coords.min()), int(z_coords.max() + 1)
                y_min, y_max = int(y_coords.min()), int(y_coords.max() + 1)
                final_z_min.append(z_min); final_z_max.append(z_max)
                final_y_min.append(y_min); final_y_max.append(y_max)
                per_slice_bboxes.append({"axis": "X", "idx": idx, "box": (z_min, z_max, y_min, y_max, x_min, x_max)})

    z_min_final = int(np.mean(final_z_min)) if final_z_min else 0
    z_max_final = int(np.mean(final_z_max)) if final_z_max else z_dim
    y_min_final = int(np.mean(final_y_min)) if final_y_min else 0
    y_max_final = int(np.mean(final_y_max)) if final_y_max else y_dim
    x_min_final = int(np.mean(final_x_min)) if final_x_min else 0
    x_max_final = int(np.mean(final_x_max)) if final_x_max else x_dim

    fused_bbox = (z_min_final, z_max_final, y_min_final, y_max_final, x_min_final, x_max_final)
    fallback_triggered = (
        z_min_final == 0 and z_max_final == z_dim and
        y_min_final == 0 and y_max_final == y_dim and
        x_min_final == 0 and x_max_final == x_dim
    )

    return fused_bbox, fallback_triggered, candidates_dict, predicted_masks_dict, per_slice_bboxes, slice_images_dict


def generate_diagnostics(case_id, raw_img, fused_bbox, predicted_masks_dict, slice_images_dict, gt_seg, cropped_resized_np):
    """
    Generate 4-panel diagnostic visualization:
    A. Original Scan (Axial, Coronal, Sagittal mid-planes)
    B. Stage-1 Vessel Predictions (Axial, Coronal, Sagittal candidate slices with mask overlay)
    C. Predicted 3D Bounding Box & Ground Truth Aneurysm Overlay
    D. Cropped & Resized 224^3 ROI (input to Stage 2)
    """
    z_min, z_max, y_min, y_max, x_min, x_max = fused_bbox
    c, z_dim, y_dim, x_dim = raw_img.shape
    vol = raw_img[0]

    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    # ---------------- PANEL A: Original Scan ----------------
    mid_z, mid_y, mid_x = z_dim // 2, y_dim // 2, x_dim // 2
    
    axes[0, 0].imshow(vol[mid_z, :, :], cmap='gray')
    axes[0, 0].set_title(f"A1: Raw Axial (Z={mid_z})", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(vol[:, mid_y, :], cmap='gray')
    axes[0, 1].set_title(f"A2: Raw Coronal (Y={mid_y})", fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(vol[:, :, mid_x], cmap='gray')
    axes[0, 2].set_title(f"A3: Raw Sagittal (X={mid_x})", fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')

    # ---------------- PANEL B: Stage-1 Vessel Predictions ----------------
    # Find candidate slices for Z, Y, X
    z_cand = [k[1] for k in predicted_masks_dict.keys() if k[0] == 'Z']
    y_cand = [k[1] for k in predicted_masks_dict.keys() if k[0] == 'Y']
    x_cand = [k[1] for k in predicted_masks_dict.keys() if k[0] == 'X']

    rep_z = z_cand[0] if z_cand else mid_z
    rep_y = y_cand[0] if y_cand else mid_y
    rep_x = x_cand[0] if x_cand else mid_x

    # Axial slice with predicted mask
    if ('Z', rep_z) in slice_images_dict:
        img_z = slice_images_dict[('Z', rep_z)]
        mask_z = predicted_masks_dict.get(('Z', rep_z), np.zeros_like(img_z))
        axes[1, 0].imshow(img_z, cmap='gray')
        if np.any(mask_z > 0):
            axes[1, 0].imshow(np.ma.masked_where(mask_z == 0, mask_z), cmap='autumn', alpha=0.6)
        axes[1, 0].set_title(f"B1: Stage 1 Pred Axial (Z={rep_z})", fontsize=11, fontweight='bold')
    else:
        axes[1, 0].imshow(vol[rep_z, :, :], cmap='gray')
        axes[1, 0].set_title(f"B1: Axial (Z={rep_z}) [No Mask]", fontsize=11)
    axes[1, 0].axis('off')

    # Coronal slice with predicted mask
    if ('Y', rep_y) in slice_images_dict:
        img_y = slice_images_dict[('Y', rep_y)]
        mask_y = predicted_masks_dict.get(('Y', rep_y), np.zeros_like(img_y))
        axes[1, 1].imshow(img_y, cmap='gray')
        if np.any(mask_y > 0):
            axes[1, 1].imshow(np.ma.masked_where(mask_y == 0, mask_y), cmap='autumn', alpha=0.6)
        axes[1, 1].set_title(f"B2: Stage 1 Pred Coronal (Y={rep_y})", fontsize=11, fontweight='bold')
    else:
        axes[1, 1].imshow(vol[:, rep_y, :], cmap='gray')
        axes[1, 1].set_title(f"B2: Coronal (Y={rep_y}) [No Mask]", fontsize=11)
    axes[1, 1].axis('off')

    # Sagittal slice with predicted mask
    if ('X', rep_x) in slice_images_dict:
        img_x = slice_images_dict[('X', rep_x)]
        mask_x = predicted_masks_dict.get(('X', rep_x), np.zeros_like(img_x))
        axes[1, 2].imshow(img_x, cmap='gray')
        if np.any(mask_x > 0):
            axes[1, 2].imshow(np.ma.masked_where(mask_x == 0, mask_x), cmap='autumn', alpha=0.6)
        axes[1, 2].set_title(f"B3: Stage 1 Pred Sagittal (X={rep_x})", fontsize=11, fontweight='bold')
    else:
        axes[1, 2].imshow(vol[:, :, rep_x], cmap='gray')
        axes[1, 2].set_title(f"B3: Sagittal (X={rep_x}) [No Mask]", fontsize=11)
    axes[1, 2].axis('off')

    # ---------------- PANEL C: Predicted 3D Bounding Box & GT Overlay ----------------
    # Axial center of bbox
    bbox_mid_z = (z_min + z_max) // 2
    bbox_mid_y = (y_min + y_max) // 2
    bbox_mid_x = (x_min + x_max) // 2

    # C1: Axial with bbox rectangle (x_min, y_min to x_max, y_max)
    axes[2, 0].imshow(vol[bbox_mid_z, :, :], cmap='gray')
    rect_z = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               linewidth=2, edgecolor='lime', facecolor='none', label='P3 ROI BBox')
    axes[2, 0].add_patch(rect_z)
    if gt_seg is not None and np.any(gt_seg[0, bbox_mid_z, :, :] > 0):
        gt_slice = gt_seg[0, bbox_mid_z, :, :]
        axes[2, 0].imshow(np.ma.masked_where(gt_slice == 0, gt_slice), cmap='spring', alpha=0.8)
    axes[2, 0].set_title(f"C1: BBox on Axial (Z={bbox_mid_z})", fontsize=11, fontweight='bold')
    axes[2, 0].axis('off')

    # C2: Coronal with bbox rectangle (x_min, z_min to x_max, z_max)
    axes[2, 1].imshow(vol[:, bbox_mid_y, :], cmap='gray')
    rect_y = patches.Rectangle((x_min, z_min), x_max - x_min, z_max - z_min,
                               linewidth=2, edgecolor='lime', facecolor='none', label='P3 ROI BBox')
    axes[2, 1].add_patch(rect_y)
    if gt_seg is not None and np.any(gt_seg[0, :, bbox_mid_y, :] > 0):
        gt_slice = gt_seg[0, :, bbox_mid_y, :]
        axes[2, 1].imshow(np.ma.masked_where(gt_slice == 0, gt_slice), cmap='spring', alpha=0.8)
    axes[2, 1].set_title(f"C2: BBox on Coronal (Y={bbox_mid_y})", fontsize=11, fontweight='bold')
    axes[2, 1].axis('off')

    # C3: Sagittal with bbox rectangle (y_min, z_min to y_max, z_max)
    axes[2, 2].imshow(vol[:, :, bbox_mid_x], cmap='gray')
    rect_x = patches.Rectangle((y_min, z_min), y_max - y_min, z_max - z_min,
                               linewidth=2, edgecolor='lime', facecolor='none', label='P3 ROI BBox')
    axes[2, 2].add_patch(rect_x)
    if gt_seg is not None and np.any(gt_seg[0, :, :, bbox_mid_x] > 0):
        gt_slice = gt_seg[0, :, :, bbox_mid_x]
        axes[2, 2].imshow(np.ma.masked_where(gt_slice == 0, gt_slice), cmap='spring', alpha=0.8)
    axes[2, 2].set_title(f"C3: BBox on Sagittal (X={bbox_mid_x})", fontsize=11, fontweight='bold')
    axes[2, 2].axis('off')

    # ---------------- PANEL D: Cropped & Resized 224^3 ROI (Stage-2 Input) ----------------
    roi_vol = cropped_resized_np.squeeze()
    r_mid_z, r_mid_y, r_mid_x = roi_vol.shape[0] // 2, roi_vol.shape[1] // 2, roi_vol.shape[2] // 2

    axes[3, 0].imshow(roi_vol[r_mid_z, :, :], cmap='gray')
    axes[3, 0].set_title(f"D1: Stage-2 Input Axial (224^3)", fontsize=11, fontweight='bold')
    axes[3, 0].axis('off')

    axes[3, 1].imshow(roi_vol[:, r_mid_y, :], cmap='gray')
    axes[3, 1].set_title(f"D2: Stage-2 Input Coronal (224^3)", fontsize=11, fontweight='bold')
    axes[3, 1].axis('off')

    axes[3, 2].imshow(roi_vol[:, :, r_mid_x], cmap='gray')
    axes[3, 2].set_title(f"D3: Stage-2 Input Sagittal (224^3)", fontsize=11, fontweight='bold')
    axes[3, 2].axis('off')

    plt.suptitle(f"P3 Stage-1 ROI Extraction & Stage-2 Input Diagnostic — {case_id}\n"
                 f"Predicted BBox: Z[{z_min}:{z_max}], Y[{y_min}:{y_max}], X[{x_min}:{x_max}] (Size: {z_max-z_min}x{y_max-y_min}x{x_max-x_min})",
                 fontsize=13, fontweight='bold', y=0.99)

    out_path = os.path.join(VIS_DIR, f"{case_id}_stage1_roi_diagnostic.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic visualization saved to: {out_path}")
    return out_path


def main():
    print("=" * 70)
    print("P3 STAGE-1 TO STAGE-2 END-TO-END BASELINE VALIDATION ON TOPANEU")
    print("=" * 70)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Load Stage 1 Predictor
    predictor = load_stage1_predictor(device)

    # Load Stage 2 Model
    stage2_model = load_stage2_model(device)

    # Load Baseline CSV for comparison
    baseline_records = {}
    if os.path.exists(BASELINE_CSV):
        with open(BASELINE_CSV, 'r') as bf:
            reader = csv.DictReader(bf)
            for r in reader:
                baseline_records[r['case_id']] = r

    results = []
    target_spacing = np.array([1, 0.55, 0.5])  # Official P3 Stage-1 target spacing

    for target in TARGET_CASES:
        case_id = target['case_id']
        is_pos = target['is_positive']
        print("\n" + "-" * 60)
        print(f"Processing case: {case_id} (Expected {'POSITIVE' if is_pos else 'NEGATIVE'})")
        print("-" * 60)

        img_path = os.path.join(TOPANEU_DIR, "images", f"{case_id}_0000.nii.gz")
        json_path = os.path.join(TOPANEU_DIR, "location_jsons", f"{case_id}.json")
        loc_mask_path = os.path.join(TOPANEU_DIR, "location_masks", f"{case_id}.nii.gz")

        # Ground truth metadata
        if os.path.exists(json_path):
            with open(json_path, 'r') as jf:
                jdata = json.load(jf)
            gt_locs = jdata.get("locations", [])
        else:
            gt_locs = []
        gt_presence = 1 if len(gt_locs) > 0 else 0
        gt_modality = "MRA"

        # Read image using SimpleITKIO
        raw_img, props = SimpleITKIO().read_images([img_path])
        orig_spacing = np.array(props['spacing'])
        c, z_dim, y_dim, x_dim = raw_img.shape
        print(f"Input Shape: {raw_img.shape}, Spacing: {orig_spacing}")

        # Ground truth segmentation mask if positive
        gt_seg = None
        gt_aneurysm_voxels = 0
        gt_coords = None
        if os.path.exists(loc_mask_path):
            gt_seg, _ = SimpleITKIO().read_seg(loc_mask_path)
            aneurysm_indices = np.where(gt_seg > 0)
            gt_aneurysm_voxels = len(aneurysm_indices[0])
            if gt_aneurysm_voxels > 0:
                gt_coords = {
                    "z_min": int(aneurysm_indices[1].min()),
                    "z_max": int(aneurysm_indices[1].max() + 1),
                    "y_min": int(aneurysm_indices[2].min()),
                    "y_max": int(aneurysm_indices[2].max() + 1),
                    "x_min": int(aneurysm_indices[3].min()),
                    "x_max": int(aneurysm_indices[3].max() + 1),
                }
                print(f"Ground Truth Aneurysm: {gt_aneurysm_voxels} voxels | BBox Z[{gt_coords['z_min']}:{gt_coords['z_max']}], Y[{gt_coords['y_min']}:{gt_coords['y_max']}], X[{gt_coords['x_min']}:{gt_coords['x_max']}]")

        # ---------------- STAGE 1 INFERENCE ----------------
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        t0_s1 = time.time()
        fused_bbox, fallback_triggered, candidates_dict, predicted_masks_dict, per_slice_bboxes, slice_images_dict = run_stage1_instrumented(
            predictor, raw_img, orig_spacing, target_spacing, max_batch_size=16
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        s1_time = time.time() - t0_s1
        s1_peak_vram = (torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == 'cuda' else 0.0

        z_min, z_max, y_min, y_max, x_min, x_max = fused_bbox
        roi_shape = (z_max - z_min, y_max - y_min, x_max - x_min)
        print(f"Stage 1 Completed in {s1_time:.3f}s | Peak VRAM: {s1_peak_vram:.1f} MB")
        print(f"Candidates: Z={candidates_dict['Z']}, Y={candidates_dict['Y']}, X={candidates_dict['X']}")
        print(f"Fused BBox: {fused_bbox} (Shape: {roi_shape}) | Fallback Triggered: {fallback_triggered}")

        # Containment Check
        is_contained = False
        contained_voxels = 0
        containment_ratio = 0.0
        if gt_aneurysm_voxels > 0 and gt_coords is not None:
            # Check overlap between GT mask and predicted ROI bbox
            roi_mask = np.zeros_like(gt_seg, dtype=bool)
            roi_mask[0, z_min:z_max, y_min:y_max, x_min:x_max] = True
            intersection = np.logical_and(gt_seg > 0, roi_mask)
            contained_voxels = int(np.sum(intersection))
            containment_ratio = contained_voxels / float(gt_aneurysm_voxels)
            is_contained = (contained_voxels > 0)
            print(f"Aneurysm Containment: {contained_voxels}/{gt_aneurysm_voxels} voxels ({containment_ratio * 100:.1f}%) -> Contained: {is_contained}")
        elif not is_pos:
            is_contained = True  # Negative case has no aneurysm to miss
            containment_ratio = 1.0

        # ---------------- STAGE 1 ROI EXTRACTION & RESIZE ----------------
        # Official notebook crop & preprocessing:
        # img_cropped_np = input_img_np[0][z_min_final:z_max_final, y_min_final:y_max_final, x_min_final:x_max_final][None]
        img_cropped_np = raw_img[0][z_min:z_max, y_min:y_max, x_min:x_max][None]
        img_cropped_np = np.ascontiguousarray(img_cropped_np)

        img_cropped_tensor = torch.from_numpy(img_cropped_np).half().to(device)
        image_normed = (img_cropped_tensor - img_cropped_tensor.mean()) / (img_cropped_tensor.std().clamp(min=1e-8))

        dst_shape = [224, 224, 224]
        image_resized = F.interpolate(
            image_normed[None], size=dst_shape, mode='trilinear', align_corners=True
        )

        cropped_resized_np = image_resized.squeeze(0).cpu().float().numpy()

        # Generate Diagnostic Figures
        diag_fig_path = generate_diagnostics(
            case_id, raw_img, fused_bbox, predicted_masks_dict, slice_images_dict, gt_seg, cropped_resized_np
        )

        # ---------------- STAGE 2 FORWARD PASS ----------------
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        t0_s2 = time.time()
        with torch.no_grad(), torch.autocast("cuda", enabled=(device.type == 'cuda')):
            conv_enc_outputs = []
            inp = image_resized.float()
            for b in stage2_model.conv_encoder_blocks:
                inp = b(inp)
                conv_enc_outputs.append(inp)
            lres = conv_enc_outputs[-1]

            p_logit = stage2_model.cls_head_list[0](lres)
            l_logits = stage2_model.cls_head_list[1](lres)
            m_logits = stage2_model.cls_modality_head(lres)

            p_prob = float(torch.sigmoid(p_logit).item())
            l_probs = torch.sigmoid(l_logits).squeeze().cpu().numpy()
            m_probs = torch.softmax(m_logits, dim=-1).squeeze().cpu().numpy()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        s2_time = time.time() - t0_s2
        s2_peak_vram = (torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == 'cuda' else 0.0

        top_loc_idx = int(np.argmax(l_probs))
        pred_loc_name = P3_LOC_NAMES[top_loc_idx]
        top_mod_idx = int(np.argmax(m_probs))
        pred_mod_name = MODALITY_NAMES[top_mod_idx]
        pred_label = 1 if p_prob >= 0.5 else 0

        print(f"Stage 2 Completed in {s2_time:.3f}s | Peak VRAM: {s2_peak_vram:.1f} MB")
        print(f"Stage-2 Predictions: Presence={p_prob:.4f} (Label={pred_label}), Top Loc={pred_loc_name} ({l_probs[top_loc_idx]:.4f}), Modality={pred_mod_name}")

        # Baseline Comparison
        b_rec = baseline_records.get(case_id, {})
        b_presence_prob = float(b_rec.get('predicted_presence_probability', -1))
        b_pred_loc = b_rec.get('predicted_location', 'N/A')
        b_inf_time = float(b_rec.get('inference_time', -1))

        print(f"Comparison with Baseline: Stage-2 Presence Prob: {b_presence_prob:.4f} (Baseline) vs {p_prob:.4f} (Stage-1 P3 ROI)")
        print(f"Comparison with Baseline: Predicted Location: '{b_pred_loc}' (Baseline) vs '{pred_loc_name}' (Stage-1 P3 ROI)")

        record = {
            "case_id": case_id,
            "modality": "MRA",
            "ground_truth_presence": gt_presence,
            "ground_truth_location": str(gt_locs),
            "input_shape": str(raw_img.shape),
            "original_spacing": f"[{orig_spacing[0]:.4f}, {orig_spacing[1]:.4f}, {orig_spacing[2]:.4f}]",
            "axial_candidates": str(candidates_dict['Z']),
            "coronal_candidates": str(candidates_dict['Y']),
            "sagittal_candidates": str(candidates_dict['X']),
            "fused_bbox": f"({z_min}, {z_max}, {y_min}, {y_max}, {x_min}, {x_max})",
            "roi_dims_before_resize": f"({roi_shape[0]}, {roi_shape[1]}, {roi_shape[2]})",
            "roi_dims_after_resize": "(224, 224, 224)",
            "fallback_triggered": fallback_triggered,
            "stage1_inference_time": round(s1_time, 4),
            "stage1_peak_vram_mb": round(s1_peak_vram, 2),
            "stage2_presence_prob": round(p_prob, 6),
            "stage2_predicted_label": pred_label,
            "stage2_predicted_location": pred_loc_name,
            "stage2_predicted_modality": pred_mod_name,
            "stage2_inference_time": round(s2_time, 4),
            "stage2_peak_vram_mb": round(s2_peak_vram, 2),
            "gt_aneurysm_voxels": gt_aneurysm_voxels,
            "contained_voxels": contained_voxels,
            "containment_ratio": round(containment_ratio, 4),
            "aneurysm_contained": is_contained,
            "baseline_presence_prob": b_presence_prob,
            "baseline_predicted_location": b_pred_loc,
            "baseline_inference_time": b_inf_time,
            "visualization_path": diag_fig_path
        }
        results.append(record)

    # Save to CSV
    fieldnames = list(results[0].keys())
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 70)
    print(f"Results successfully exported to: {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == '__main__':
    main()
