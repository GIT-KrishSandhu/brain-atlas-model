import os
import sys
import json
import time
import csv
import numpy as np
import pandas as pd
from scipy.ndimage import zoom, label, center_of_mass
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Configure module paths
sys.path.insert(0, r'D:\NLP_Project\scratch\dna_src\dynamic_network_architectures-0.3.1')
sys.path.insert(0, r'D:\NLP_Project\RSNA2025_Intracranial-Aneurysm-Detection\nnXNet')

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet_two_seg_with_cls_modality import ResEncoderUNet_two_seg_with_cls_modality

TOPANEU_DIR = r"D:\NLP_Project\topaneu_release"
STAGE2_CKPT = r"D:\NLP_Project\scratch\checkpoints\Dataset660_26classes_resize224_4661\onlyMirror01_lr4e3_100epochs_ps224\fold_0\checkpoint_final.pth"
BASELINE_CSV = r"D:\NLP_Project\p3_pretrained_topaneu_predictions.csv"
MAPPING_CSV = r"D:\NLP_Project\PHASE2_LABEL_MAPPING.csv"

PRED_DIR = r"D:\NLP_Project\scratch\phase2_segmentation_predictions"
VIS_DIR = r"D:\NLP_Project\scratch\phase2_visualizations"
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

OUT_SEG_METRICS_CSV = r"D:\NLP_Project\phase2_segmentation_metrics.csv"
OUT_LOC_ALIGN_CSV = r"D:\NLP_Project\phase2_location_alignment.csv"

P3_LOC_NAMES = [
    "Other Posterior Circulation",                  # 1
    "Basilar Tip",                                  # 2
    "Right Posterior Communicating Artery",         # 3
    "Left Posterior Communicating Artery",          # 4
    "Right Infraclinoid Internal Carotid Artery",   # 5
    "Left Infraclinoid Internal Carotid Artery",    # 6
    "Right Supraclinoid Internal Carotid Artery",   # 7
    "Left Supraclinoid Internal Carotid Artery",    # 8
    "Right Middle Cerebral Artery",                 # 9
    "Left Middle Cerebral Artery",                  # 10
    "Right Anterior Cerebral Artery",                # 11
    "Left Anterior Cerebral Artery",                 # 12
    "Anterior Communicating Artery"                 # 13
]

# Name normalizer for robust string comparison
def norm_name(s):
    if not isinstance(s, str): return ""
    s = s.lower().strip()
    s = s.replace("internal carotid artery", "ica")
    s = s.replace("posterior communicating artery", "pcom")
    s = s.replace("middle cerebral artery", "mca")
    s = s.replace("anterior cerebral artery", "aca")
    s = s.replace("ba-tip", "basilar tip")
    s = s.replace("anterior communicating artery", "acom")
    s = s.replace("acom complex", "acom")
    return s


def compute_binary_metrics(pred_bin, gt_bin):
    tp = np.count_nonzero(pred_bin & gt_bin)
    fp = np.count_nonzero(pred_bin & ~gt_bin)
    fn = np.count_nonzero(~pred_bin & gt_bin)
    tn = np.count_nonzero(~pred_bin & ~gt_bin)
    support = tp + fn

    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else (1.0 if support == 0 and fp == 0 else 0.0)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else (1.0 if support == 0 and fp == 0 else 0.0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if support == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if support == 0 else 0.0)

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "support": int(support),
        "dice": float(dice), "iou": float(iou),
        "precision": float(precision), "recall": float(recall)
    }


def load_model(device):
    print("Initializing ResEncoderUNet_two_seg_with_cls_modality...")
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

    print(f"Loading weights from: {STAGE2_CKPT}...")
    ckpt = torch.load(STAGE2_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["network_weights"], strict=True)
    model.eval()
    print("Model loaded strictly!")
    return model


def run_part8_location_alignment():
    print("\n" + "=" * 70)
    print("PART 8: P3 LOCATION PREDICTION VS GROUND TRUTH (ALL 415 CASES)")
    print("=" * 70)

    df = pd.read_csv(BASELINE_CSV)
    mapping_df = pd.read_csv(MAPPING_CSV)
    loc_rules = mapping_df[mapping_df["category"] == "Location (Aneurysm)"].set_index("topaneu_id")
    vess_rules = mapping_df[mapping_df["category"] == "Vessel (Anatomy)"].set_index("topaneu_id")

    with open(os.path.join(TOPANEU_DIR, "location_mapping.json")) as f:
        loc_map = json.load(f)["labels"]
    inv_loc = {v: k for k, v in loc_map.items()}

    records = []
    correct_count = 0
    pos_cases_count = 0

    per_class_stats = {name: {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "pred_count": 0} for name in P3_LOC_NAMES}

    for _, row in df.iterrows():
        cid = row["case_id"]
        gt_presence = row["ground_truth_presence"]
        pred_loc = str(row["predicted_location"])
        norm_pred = norm_name(pred_loc)

        if gt_presence == 1:
            pos_cases_count += 1
            gt_loc_ids = eval(str(row["ground_truth_location"]))
            gt_names = [inv_loc.get(gid, "Unknown") for gid in gt_loc_ids]
            
            mapped_p3_names = []
            for gid in gt_loc_ids:
                if gid in loc_rules.index:
                    mapped_p3_names.append(loc_rules.loc[gid, "p3_name"])
                else:
                    mapped_p3_names.append("NO_DIRECT_EQUIVALENT")

            norm_mapped = [norm_name(m) for m in mapped_p3_names]
            is_correct = any(norm_pred == m for m in norm_mapped)
            if is_correct:
                correct_count += 1

            # Vessel association
            gt_vessels = []
            for gn in gt_names:
                if "ICA" in gn: gt_vessels.append("ICA")
                elif "MCA" in gn or "M1" in gn or "M2" in gn: gt_vessels.append("MCA")
                elif "ACA" in gn or "A1" in gn or "A2" in gn or "A3" in gn: gt_vessels.append("ACA")
                elif "Acom" in gn: gt_vessels.append("ACom")
                elif "Pcom" in gn: gt_vessels.append("PCom")
                elif "BA tip" in gn or "BA" in gn: gt_vessels.append("Basilar")
                elif "VA" in gn or "PICA" in gn or "AICA" in gn or "SCA" in gn or "P1P2" in gn: gt_vessels.append("Posterior Circulation")
                else: gt_vessels.append("Other")

            pred_vessel = "Other"
            if "ICA" in pred_loc: pred_vessel = "ICA"
            elif "MCA" in pred_loc: pred_vessel = "MCA"
            elif "ACA" in pred_loc: pred_vessel = "ACA"
            elif "Acom" in pred_loc or "Communicating Artery" in pred_loc and "Anterior" in pred_loc: pred_vessel = "ACom"
            elif "Pcom" in pred_loc: pred_vessel = "PCom"
            elif "Basilar" in pred_loc: pred_vessel = "Basilar"
            elif "Posterior" in pred_loc: pred_vessel = "Posterior Circulation"

            vessel_aligned = any(pred_vessel == gtv for gtv in gt_vessels)
            status = "EXACT_LOCATION_MATCH" if is_correct else ("VESSEL_ALIGNED_ONLY" if vessel_aligned else "MISALIGNED")

            # Per class stats
            for p3_name in P3_LOC_NAMES:
                norm_p3 = norm_name(p3_name)
                is_gt_class = any(norm_p3 == m for m in norm_mapped)
                is_pred_class = (norm_pred == norm_p3)

                if is_gt_class: per_class_stats[p3_name]["gt_count"] += 1
                if is_pred_class: per_class_stats[p3_name]["pred_count"] += 1

                if is_gt_class and is_pred_class:
                    per_class_stats[p3_name]["tp"] += 1
                elif is_pred_class and not is_gt_class:
                    per_class_stats[p3_name]["fp"] += 1
                elif is_gt_class and not is_pred_class:
                    per_class_stats[p3_name]["fn"] += 1

            records.append({
                "case_id": cid,
                "ground_truth_location": "; ".join(gt_names),
                "predicted_location": pred_loc,
                "classification_correct": is_correct,
                "ground_truth_vessel": "; ".join(set(gt_vessels)),
                "predicted_vessel": pred_vessel,
                "spatial_alignment_status": status,
                "notes": f"Mapped P3 GT: {'; '.join(set(mapped_p3_names))}"
            })
        else:
            records.append({
                "case_id": cid,
                "ground_truth_location": "NEGATIVE",
                "predicted_location": pred_loc,
                "classification_correct": True,
                "ground_truth_vessel": "NONE",
                "predicted_vessel": "N/A (Normal)",
                "spatial_alignment_status": "TRUE_NEGATIVE",
                "notes": "Non-aneurysm case"
            })

    # Save CSV
    loc_df = pd.DataFrame(records)
    loc_df.to_csv(OUT_LOC_ALIGN_CSV, index=False)
    print(f"Location alignment records saved to: {OUT_LOC_ALIGN_CSV}")

    top1_acc = correct_count / pos_cases_count if pos_cases_count > 0 else 0
    print(f"Top-1 Location Accuracy on Positive Cases: {correct_count}/{pos_cases_count} = {top1_acc*100:.2f}%")

    # Per-class metrics
    class_metrics = []
    for p3_name in P3_LOC_NAMES:
        st = per_class_stats[p3_name]
        tp, fp, fn, gt_c = st["tp"], st["fp"], st["fn"], st["gt_count"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        class_metrics.append({
            "class_name": p3_name,
            "gt_support": gt_c,
            "pred_count": st["pred_count"],
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1
        })

    class_metrics_df = pd.DataFrame(class_metrics)
    macro_f1 = class_metrics_df["f1"].mean()
    print(f"Macro F1 for 13 Location Classes: {macro_f1:.4f}")
    return top1_acc, macro_f1, class_metrics_df


def select_stratified_cohort():
    """Select a stratified representative cohort of 40 cases covering all 13 locations + focal cases + negatives"""
    df = pd.read_csv(BASELINE_CSV)
    mapping_df = pd.read_csv(MAPPING_CSV)
    loc_rules = mapping_df[mapping_df["category"] == "Location (Aneurysm)"].set_index("topaneu_id")

    with open(os.path.join(TOPANEU_DIR, "location_mapping.json")) as f:
        loc_map = json.load(f)["labels"]
    inv_loc = {v: k for k, v in loc_map.items()}

    focal_positives = ["topaneu_center1_mr_017", "topaneu_center1_mr_024", "topaneu_center1_mr_028"]
    focal_negatives = ["topaneu_center1_mr_001"]

    selected = list(focal_positives) + list(focal_negatives)

    # Bucket positive cases by mapped P3 territory
    pos_df = df[df["ground_truth_presence"] == 1]
    bucket = {name: [] for name in P3_LOC_NAMES}

    for _, row in pos_df.iterrows():
        cid = row["case_id"]
        if cid in selected: continue
        gids = eval(str(row["ground_truth_location"]))
        for gid in gids:
            if gid in loc_rules.index:
                p3_name = loc_rules.loc[gid, "p3_name"]
                if p3_name in bucket:
                    bucket[p3_name].append(cid)

    # Pick 2-3 cases per P3 territory
    for p3_name, cids in bucket.items():
        for cid in cids[:2]:
            if cid not in selected and len(selected) < 35:
                selected.append(cid)

    # Add 5 more negative cases
    neg_df = df[df["ground_truth_presence"] == 0]
    for _, row in neg_df.iterrows():
        cid = row["case_id"]
        if cid not in selected and len(selected) < 40:
            selected.append(cid)

    print(f"Selected representative stratified cohort of {len(selected)} cases.")
    return selected


def run_segmentation_evaluation(model, device, cohort_cases):
    print("\n" + "=" * 70)
    print(f"RUNNING P3 SEGMENTATION INFERENCE & EVALUATION ({len(cohort_cases)} CASES)")
    print("=" * 70)

    mapping_df = pd.read_csv(MAPPING_CSV)
    loc_rules = mapping_df[mapping_df["category"] == "Location (Aneurysm)"].set_index("topaneu_id")
    vess_rules = mapping_df[mapping_df["category"] == "Vessel (Anatomy)"].set_index("topaneu_id")

    global_vessel_stats = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    global_aneurysm_stats = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    # Per-class territory stats (13 P3 classes)
    per_class_territory_stats = {name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "support": 0} for name in P3_LOC_NAMES}

    # Lesion level accumulators
    gt_lesions_total = 0
    detected_lesions_total = 0
    fp_lesions_total = 0
    centroid_distances = []

    focal_visualization_data = {}

    for idx, cid in enumerate(cohort_cases, 1):
        t0 = time.time()
        img_p = os.path.join(TOPANEU_DIR, "images", f"{cid}_0000.nii.gz")
        vess_p = os.path.join(TOPANEU_DIR, "vessel_masks", f"{cid}.nii.gz")
        loc_p = os.path.join(TOPANEU_DIR, "location_masks", f"{cid}.nii.gz")

        # 1. Load Raw Volume
        nii_img = nib.load(img_p)
        raw_vol = nii_img.get_fdata().astype(np.float32)
        factors = [224.0 / s for s in raw_vol.shape]

        # 2. Resample image via trilinear interpolation and normalize
        vol_tensor = torch.from_numpy(raw_vol).unsqueeze(0).unsqueeze(0).to(device)
        vol_224 = F.interpolate(vol_tensor, size=(224, 224, 224), mode='trilinear', align_corners=True)
        vol_norm = (vol_224 - vol_224.mean()) / vol_224.std().clamp(min=1e-8)

        # 3. Resample Ground Truth Masks via Nearest-Neighbor Interpolation
        gt_vess_data = nib.load(vess_p).get_fdata()
        gt_loc_data = nib.load(loc_p).get_fdata()

        vess_tensor = torch.from_numpy(gt_vess_data).unsqueeze(0).unsqueeze(0).float().to(device)
        gt_vess_224 = F.interpolate(vess_tensor, size=(224, 224, 224), mode='nearest').squeeze().cpu().numpy().astype(np.uint8)

        loc_tensor = torch.from_numpy(gt_loc_data).unsqueeze(0).unsqueeze(0).float().to(device)
        gt_loc_224 = F.interpolate(loc_tensor, size=(224, 224, 224), mode='nearest').squeeze().cpu().numpy().astype(np.uint8)

        gt_aneurysm_bin = (gt_loc_224 > 0)
        gt_vessel_bin = (gt_vess_224 > 0)

        # 4. Forward Pass through P3 Dual Decoders
        with torch.no_grad(), torch.autocast('cuda', enabled=(device.type == 'cuda')):
            r1, r2, cls_list, mod_pred = model(vol_norm, only_forward_cls=False)

        pred_seg1 = r1.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        pred_seg2 = r2.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

        # Decoder 1:
        # Channels 1..13 are normal vessels; Channel 14 is binary aneurysm
        pred_aneurysm_bin = (pred_seg1 == 14)
        pred_vessel_bin = ((pred_seg1 >= 1) & (pred_seg1 <= 13))

        # Combined 26-class prediction map
        combined_26 = np.zeros_like(pred_seg1, dtype=np.uint8)
        combined_26[pred_vessel_bin] = pred_seg1[pred_vessel_bin]
        aneurysm_mask = pred_aneurysm_bin & (pred_seg2 >= 1) & (pred_seg2 <= 13)
        combined_26[aneurysm_mask] = (13 + pred_seg2[aneurysm_mask]).astype(np.uint8)

        # Save compact .npz
        npz_out = os.path.join(PRED_DIR, f"{cid}.npz")
        np.savez_compressed(
            npz_out,
            seg1_vessel=pred_seg1,
            seg2_territory=pred_seg2,
            combined_26=combined_26
        )

        # 5. Calculate Metrics for this Case
        # Vessel Metrics
        v_m = compute_binary_metrics(pred_vessel_bin, gt_vessel_bin)
        for k in ["tp", "fp", "fn", "tn"]: global_vessel_stats[k] += v_m[k]

        # Aneurysm Metrics
        a_m = compute_binary_metrics(pred_aneurysm_bin, gt_aneurysm_bin)
        for k in ["tp", "fp", "fn", "tn"]: global_aneurysm_stats[k] += a_m[k]

        # Per-Class Territory Evaluation (seg_layers_2 vs mapped GT)
        for p3_idx, p3_name in enumerate(P3_LOC_NAMES, 1):
            pred_territory_bin = (pred_seg2 == p3_idx)
            # Find which TopAneu vessel IDs map to this P3 territory
            mapped_vess_ids = [tid for tid, row in vess_rules.iterrows() if row["p3_name"] == p3_name]
            mapped_loc_ids = [tid for tid, row in loc_rules.iterrows() if row["p3_name"] == p3_name]

            gt_territory_bin = np.isin(gt_vess_224, mapped_vess_ids) | np.isin(gt_loc_224, mapped_loc_ids)
            t_m = compute_binary_metrics(pred_territory_bin, gt_territory_bin)

            for k in ["tp", "fp", "fn", "tn"]:
                per_class_territory_stats[p3_name][k] += t_m[k]
            per_class_territory_stats[p3_name]["support"] += t_m["support"]

        # 6. Lesion-Level Evaluation (for positive cases)
        if np.any(gt_aneurysm_bin):
            labeled_gt, num_gt = label(gt_aneurysm_bin)
            labeled_pred, num_pred = label(pred_aneurysm_bin)
            gt_lesions_total += num_gt

            pred_matched = set()
            for l_idx in range(1, num_gt + 1):
                lesion_mask = (labeled_gt == l_idx)
                # Check overlap with predicted lesions
                overlap = np.logical_and(lesion_mask, pred_aneurysm_bin)
                iou_lesion = np.count_nonzero(overlap) / float(np.count_nonzero(np.logical_or(lesion_mask, pred_aneurysm_bin))) if np.any(overlap) else 0.0

                if np.any(overlap) and iou_lesion > 0.01:
                    detected_lesions_total += 1
                    # Centroid distance
                    gt_cm = center_of_mass(lesion_mask)
                    # Find closest predicted component
                    overlapping_pred_labels = np.unique(labeled_pred[overlap])
                    overlapping_pred_labels = overlapping_pred_labels[overlapping_pred_labels > 0]
                    if len(overlapping_pred_labels) > 0:
                        pred_cm = center_of_mass(labeled_pred == overlapping_pred_labels[0])
                        dist_voxels = np.linalg.norm(np.array(gt_cm) - np.array(pred_cm))
                        centroid_distances.append(dist_voxels)
                        pred_matched.add(overlapping_pred_labels[0])

            fp_lesions_total += max(0, num_pred - len(pred_matched))

        # Store data for focal visualizations
        if cid in ["topaneu_center1_mr_017", "topaneu_center1_mr_024", "topaneu_center1_mr_028", "topaneu_center1_mr_001"]:
            focal_visualization_data[cid] = {
                "vol_224": vol_224.squeeze().cpu().numpy(),
                "gt_vess_224": gt_vess_224,
                "gt_loc_224": gt_loc_224,
                "pred_seg1": pred_seg1,
                "pred_seg2": pred_seg2,
                "pred_vessel_bin": pred_vessel_bin,
                "pred_aneurysm_bin": pred_aneurysm_bin
            }

        dur = time.time() - t0
        print(f"[{idx:02d}/{len(cohort_cases):02d}] {cid:24s} | Vessel Dice: {v_m['dice']:.3f} | Aneurysm Dice: {a_m['dice']:.3f} | Time: {dur:.2f}s")

    # Compute Global & Per-Class Aggregates
    v_tp, v_fp, v_fn = global_vessel_stats["tp"], global_vessel_stats["fp"], global_vessel_stats["fn"]
    global_vessel_dice = (2.0 * v_tp) / (2.0 * v_tp + v_fp + v_fn) if (2 * v_tp + v_fp + v_fn) > 0 else 0.0
    global_vessel_iou = v_tp / (v_tp + v_fp + v_fn) if (v_tp + v_fp + v_fn) > 0 else 0.0
    global_vessel_prec = v_tp / (v_tp + v_fp) if (v_tp + v_fp) > 0 else 0.0
    global_vessel_rec = v_tp / (v_tp + v_fn) if (v_tp + v_fn) > 0 else 0.0

    a_tp, a_fp, a_fn = global_aneurysm_stats["tp"], global_aneurysm_stats["fp"], global_aneurysm_stats["fn"]
    global_aneurysm_dice = (2.0 * a_tp) / (2.0 * a_tp + a_fp + a_fn) if (2 * a_tp + a_fp + a_fn) > 0 else 0.0
    global_aneurysm_iou = a_tp / (a_tp + a_fp + a_fn) if (a_tp + a_fp + a_fn) > 0 else 0.0
    global_aneurysm_prec = a_tp / (a_tp + a_fp) if (a_tp + a_fp) > 0 else 0.0
    global_aneurysm_rec = a_tp / (a_tp + a_fn) if (a_tp + a_fn) > 0 else 0.0

    print("\n" + "=" * 70)
    print("GLOBAL SEGMENTATION RESULTS:")
    print(f"  Vessel Segmentation:   Dice = {global_vessel_dice:.4f} | IoU = {global_vessel_iou:.4f} | Prec = {global_vessel_prec:.4f} | Rec = {global_vessel_rec:.4f}")
    print(f"  Aneurysm Segmentation: Dice = {global_aneurysm_dice:.4f} | IoU = {global_aneurysm_iou:.4f} | Prec = {global_aneurysm_prec:.4f} | Rec = {global_aneurysm_rec:.4f}")
    print("=" * 70)

    # Save to phase2_segmentation_metrics.csv
    seg_metrics_rows = [
        {
            "class_group": "GLOBAL",
            "class_name": "Vessel Segmentation (Tree)",
            "dice": round(global_vessel_dice, 4),
            "iou": round(global_vessel_iou, 4),
            "precision": round(global_vessel_prec, 4),
            "recall": round(global_vessel_rec, 4),
            "support_voxels": int(global_vessel_stats["tp"] + global_vessel_stats["fn"])
        },
        {
            "class_group": "GLOBAL",
            "class_name": "Aneurysm Lesion Segmentation",
            "dice": round(global_aneurysm_dice, 4),
            "iou": round(global_aneurysm_iou, 4),
            "precision": round(global_aneurysm_prec, 4),
            "recall": round(global_aneurysm_rec, 4),
            "support_voxels": int(global_aneurysm_stats["tp"] + global_aneurysm_stats["fn"])
        }
    ]

    territory_dices = []
    territory_ious = []
    for p3_name in P3_LOC_NAMES:
        st = per_class_territory_stats[p3_name]
        tp, fp, fn, supp = st["tp"], st["fp"], st["fn"], st["support"]
        dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else (1.0 if supp == 0 and fp == 0 else 0.0)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else (1.0 if supp == 0 and fp == 0 else 0.0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        territory_dices.append(dice)
        territory_ious.append(iou)
        seg_metrics_rows.append({
            "class_group": "TERRITORY",
            "class_name": p3_name,
            "dice": round(dice, 4),
            "iou": round(iou, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "support_voxels": int(supp)
        })

    # Add Macro Average
    seg_metrics_rows.append({
        "class_group": "SUMMARY",
        "class_name": "Macro Average (13 Territories)",
        "dice": round(float(np.mean(territory_dices)), 4),
        "iou": round(float(np.mean(territory_ious)), 4),
        "precision": round(float(np.mean([r["precision"] for r in seg_metrics_rows if r["class_group"] == "TERRITORY"])), 4),
        "recall": round(float(np.mean([r["recall"] for r in seg_metrics_rows if r["class_group"] == "TERRITORY"])), 4),
        "support_voxels": sum([r["support_voxels"] for r in seg_metrics_rows if r["class_group"] == "TERRITORY"])
    })

    pd.DataFrame(seg_metrics_rows).to_csv(OUT_SEG_METRICS_CSV, index=False)
    print(f"Saved segmentation metrics to: {OUT_SEG_METRICS_CSV}")

    # Lesion metrics summary
    mean_cd = float(np.mean(centroid_distances)) if centroid_distances else 0.0
    lesion_det_rate = (detected_lesions_total / gt_lesions_total) if gt_lesions_total > 0 else 0.0

    return {
        "global_vessel_dice": global_vessel_dice, "global_vessel_iou": global_vessel_iou,
        "global_aneurysm_dice": global_aneurysm_dice, "global_aneurysm_iou": global_aneurysm_iou,
        "gt_lesions": gt_lesions_total, "detected_lesions": detected_lesions_total,
        "lesion_detection_rate": lesion_det_rate,
        "fp_lesions": fp_lesions_total,
        "mean_centroid_dist": mean_cd,
        "focal_visualization_data": focal_visualization_data,
        "seg_metrics_rows": seg_metrics_rows
    }


def generate_focal_visualizations(focal_data):
    print("\n" + "=" * 70)
    print("PART 10: GENERATING CASE-LEVEL FOCAL DIAGNOSTIC VISUALIZATIONS")
    print("=" * 70)

    for cid, data in focal_data.items():
        vol = data["vol_224"]
        gt_vess = data["gt_vess_224"]
        gt_loc = data["gt_loc_224"]
        pred_seg1 = data["pred_seg1"]
        pred_seg2 = data["pred_seg2"]
        pred_vessel_bin = data["pred_vessel_bin"]
        pred_aneurysm_bin = data["pred_aneurysm_bin"]

        # Determine the FOCAL SLICE containing the aneurysm
        if np.any(gt_loc > 0):
            # Compute center of mass of the aneurysm
            z_idx = int(np.round(center_of_mass(gt_loc > 0)[0]))
            slice_type = f"Focal Aneurysm Slice (Z={z_idx})"
        else:
            z_idx = 112
            slice_type = f"Mid-Volume Slice (Z={z_idx}) [Negative Control]"

        z_idx = np.clip(z_idx, 0, 223)

        img_slice = vol[z_idx, :, :]
        gt_v_slice = gt_vess[z_idx, :, :]
        gt_a_slice = gt_loc[z_idx, :, :]
        p_v_slice = pred_vessel_bin[z_idx, :, :]
        p_a_slice = pred_aneurysm_bin[z_idx, :, :]
        p_t_slice = pred_seg2[z_idx, :, :]

        fig, axes = plt.subplots(2, 3, figsize=(16, 11))
        plt.subplots_adjust(hspace=0.25, wspace=0.15)

        # 1. Raw Image
        axes[0, 0].imshow(img_slice, cmap='gray')
        axes[0, 0].set_title(f"1. Raw Scan ({slice_type})", fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')

        # 2. GT Vessel Mask
        axes[0, 1].imshow(img_slice, cmap='gray')
        if np.any(gt_v_slice > 0):
            axes[0, 1].imshow(np.ma.masked_where(gt_v_slice == 0, gt_v_slice), cmap='spring', alpha=0.6)
        axes[0, 1].set_title("2. Ground Truth Vessels (TopAneu)", fontsize=11, fontweight='bold')
        axes[0, 1].axis('off')

        # 3. GT Aneurysm / Location Mask
        axes[0, 2].imshow(img_slice, cmap='gray')
        if np.any(gt_a_slice > 0):
            axes[0, 2].imshow(np.ma.masked_where(gt_a_slice == 0, gt_a_slice), cmap='autumn', alpha=0.85)
            # Add annotation text
            cm = center_of_mass(gt_a_slice > 0)
            axes[0, 2].plot(cm[1], cm[0], 'ro', markersize=8)
        axes[0, 2].set_title(f"3. GT Aneurysm (Voxels: {np.count_nonzero(gt_a_slice)})", fontsize=11, fontweight='bold')
        axes[0, 2].axis('off')

        # 4. P3 Predicted Vessel Segmentation
        axes[1, 0].imshow(img_slice, cmap='gray')
        if np.any(p_v_slice > 0):
            axes[1, 0].imshow(np.ma.masked_where(p_v_slice == 0, p_v_slice), cmap='cool', alpha=0.6)
        axes[1, 0].set_title("4. P3 Predicted Vessels (seg_layers_1)", fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')

        # 5. P3 Predicted Aneurysm Segmentation
        axes[1, 1].imshow(img_slice, cmap='gray')
        if np.any(p_a_slice > 0):
            axes[1, 1].imshow(np.ma.masked_where(p_a_slice == 0, p_a_slice), cmap='hot', alpha=0.85)
            cm_p = center_of_mass(p_a_slice > 0)
            axes[1, 1].plot(cm_p[1], cm_p[0], 'mo', markersize=8)
        axes[1, 1].set_title(f"5. P3 Pred Aneurysm (Channel 14, Voxels: {np.count_nonzero(p_a_slice)})", fontsize=11, fontweight='bold')
        axes[1, 1].axis('off')

        # 6. P3 Predicted Territory (seg_layers_2)
        axes[1, 2].imshow(img_slice, cmap='gray')
        if np.any(p_t_slice > 0):
            axes[1, 2].imshow(np.ma.masked_where(p_t_slice == 0, p_t_slice), cmap='tab20', alpha=0.65)
        axes[1, 2].set_title("6. P3 Anatomical Territory (seg_layers_2)", fontsize=11, fontweight='bold')
        axes[1, 2].axis('off')

        plt.suptitle(f"Phase 2 Anatomical Grounding & Spatial Alignment — {cid}\n"
                     f"Spatial Slice: Z = {z_idx} (Strictly Matched Across All Panels)",
                     fontsize=13, fontweight='bold', y=0.98)

        out_path = os.path.join(VIS_DIR, f"{cid}_phase2_focal_diagnostic.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved focal diagnostic visualization to: {out_path}")


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Step 1: Location classification alignment on all 415 cases
    top1_acc, macro_f1, class_metrics_df = run_part8_location_alignment()

    # Step 2: Stratified cohort selection
    cohort = select_stratified_cohort()

    # Step 3: Load Model strictly
    model = load_model(device)

    # Step 4: Run segmentation evaluation & export predictions
    seg_results = run_segmentation_evaluation(model, device, cohort)

    # Step 5: Focal visualizations
    generate_focal_visualizations(seg_results["focal_visualization_data"])

    print("\n" + "=" * 70)
    print("PHASE 2 EXECUTION COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
