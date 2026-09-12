import os
import sys
import time
import json
import csv
import numpy as np

sys.path.insert(0, r'D:\NLP_Project\scratch\dna_src\dynamic_network_architectures-0.3.1')
sys.path.insert(0, r'D:\NLP_Project\RSNA2025_Intracranial-Aneurysm-Detection\nnXNet')

import torch
import torch.nn as nn
from scipy.ndimage import zoom
import nibabel as nib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)

from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet_two_seg_with_cls_modality import ResEncoderUNet_two_seg_with_cls_modality

TOPANEU_DIR = r"D:\NLP_Project\topaneu_release"
CKPT_PATH = r"D:\NLP_Project\scratch\checkpoints\Dataset660_26classes_resize224_4661\onlyMirror01_lr4e3_100epochs_ps224\fold_0\checkpoint_final.pth"
OUTPUT_CSV_1 = r"D:\NLP_Project\p3_pretrained_topaneu_predictions.csv"
OUTPUT_CSV_2 = r"D:\NLP_Project\RSNA2025_Intracranial-Aneurysm-Detection\p3_pretrained_topaneu_predictions.csv"
SUMMARY_JSON = r"D:\NLP_Project\scratch\p3_evaluation_summary.json"

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

def main():
    print("=" * 60)
    print("P3 PRETRAINED MODEL: FULL TOPANEU EVALUATION (STATE E)")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Build Model
    print("Constructing ResEncoderUNet_two_seg_with_cls_modality...")
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

    # 2. Load Checkpoint Strictly
    print(f"Loading checkpoint from: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    load_res = model.load_state_dict(ckpt["network_weights"], strict=True)
    print(f"Strict load result: {load_res}")
    model.eval()

    # 3. Discover Cases
    images_dir = os.path.join(TOPANEU_DIR, "images")
    json_dir = os.path.join(TOPANEU_DIR, "location_jsons")
    all_files = sorted([f for f in os.listdir(images_dir) if f.endswith("_0000.nii.gz") and not f.startswith("._")])
    print(f"Discovered {len(all_files)} cases in {images_dir}")

    records = []
    t_eval_start = time.time()

    for idx, fname in enumerate(all_files, 1):
        case_id = fname.replace("_0000.nii.gz", "")
        img_path = os.path.join(images_dir, fname)
        json_path = os.path.join(json_dir, f"{case_id}.json")

        # Ground Truth
        if os.path.exists(json_path):
            with open(json_path) as jf:
                jdata = json.load(jf)
            gt_locs = jdata.get("locations", [])
        else:
            gt_locs = []
        gt_presence = 1 if len(gt_locs) > 0 else 0
        gt_modality = "MRA" if "_mr_" in case_id else "CTA"

        # Preprocessing
        t0_prep = time.time()
        nii = nib.load(img_path)
        raw_vol = nii.get_fdata().astype(np.float32)

        factors = [224.0 / s for s in raw_vol.shape]
        resampled_vol = zoom(raw_vol, factors, order=1)

        # Standard P3 ZScoreNormalization
        mean = resampled_vol.mean()
        std = max(resampled_vol.std(), 1e-8)
        norm_vol = (resampled_vol - mean) / std
        t_prep = time.time() - t0_prep

        # Inference
        x_tensor = torch.from_numpy(norm_vol).unsqueeze(0).unsqueeze(0).float().to(device)

        torch.cuda.synchronize()
        t0_inf = time.time()
        with torch.no_grad(), torch.autocast("cuda", enabled=True):
            conv_enc_outputs = []
            inp = x_tensor
            for b in model.conv_encoder_blocks:
                inp = b(inp)
                conv_enc_outputs.append(inp)
            lres = conv_enc_outputs[-1]

            p_logit = model.cls_head_list[0](lres)
            l_logits = model.cls_head_list[1](lres)
            m_logits = model.cls_modality_head(lres)

            p_prob = torch.sigmoid(p_logit).item()
            l_probs = torch.sigmoid(l_logits).squeeze().cpu().numpy()
            m_probs = torch.softmax(m_logits, dim=-1).squeeze().cpu().numpy()

        torch.cuda.synchronize()
        t_inf = time.time() - t0_inf

        pred_label = 1 if p_prob >= 0.5 else 0
        top_loc_idx = int(np.argmax(l_probs))
        pred_loc_name = P3_LOC_NAMES[top_loc_idx]
        top_mod_idx = int(np.argmax(m_probs))
        pred_mod_name = MODALITY_NAMES[top_mod_idx]

        record = {
            "case_id": case_id,
            "ground_truth_presence": gt_presence,
            "predicted_presence_probability": round(p_prob, 6),
            "predicted_presence_label": pred_label,
            "ground_truth_location": str(gt_locs),
            "predicted_location": pred_loc_name,
            "location_probabilities": ";".join([f"{p:.5f}" for p in l_probs]),
            "ground_truth_modality": gt_modality,
            "predicted_modality": pred_mod_name,
            "inference_time": round(t_inf, 4),
            "preprocessing_time": round(t_prep, 4)
        }
        records.append(record)

        if idx % 25 == 0 or idx == len(all_files):
            elapsed = time.time() - t_eval_start
            speed = idx / elapsed
            rem = (len(all_files) - idx) / speed if speed > 0 else 0
            print(f"[{idx}/{len(all_files)}] Elapsed: {elapsed:.1f}s | Speed: {speed:.2f} cases/s | ETA: {rem:.1f}s | Latest Case {case_id}: GT={gt_presence}, Prob={p_prob:.4f}, Prep={t_prep:.2f}s, Inf={t_inf*1000:.0f}ms")
            sys.stdout.flush()

    # 4. Save CSV predictions
    fieldnames = [
        "case_id",
        "ground_truth_presence",
        "predicted_presence_probability",
        "predicted_presence_label",
        "ground_truth_location",
        "predicted_location",
        "location_probabilities",
        "ground_truth_modality",
        "predicted_modality",
        "inference_time",
        "preprocessing_time"
    ]

    for csv_path in [OUTPUT_CSV_1, OUTPUT_CSV_2]:
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"Predictions saved to: {csv_path}")

    # 5. Compute Metrics
    y_true = np.array([r["ground_truth_presence"] for r in records])
    y_probs = np.array([r["predicted_presence_probability"] for r in records])
    y_pred_default = np.array([r["predicted_presence_label"] for r in records])

    # Confusion matrix & Primary metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_default).ravel()
    acc = accuracy_score(y_true, y_pred_default)
    rec = recall_score(y_true, y_pred_default, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, y_pred_default, zero_division=0)
    f1 = f1_score(y_true, y_pred_default, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    brier = brier_score_loss(y_true, y_probs)

    # Modality metrics
    gt_mods = [r["ground_truth_modality"] for r in records]
    pred_mods = [r["predicted_modality"] for r in records]
    mod_acc = sum(1 for g, p in zip(gt_mods, pred_mods) if g == p) / len(records)

    # Threshold sweeps (Exploratory Analysis)
    thresholds = np.linspace(0.05, 0.95, 19)
    sweep_results = []
    for th in thresholds:
        yp = (y_probs >= th).astype(int)
        cm = confusion_matrix(y_true, yp)
        if cm.shape == (2, 2):
            t_tn, t_fp, t_fn, t_tp = cm.ravel()
        else:
            t_tn = t_fp = t_fn = t_tp = 0
        t_acc = accuracy_score(y_true, yp)
        t_rec = recall_score(y_true, yp, zero_division=0)
        t_spec = t_tn / (t_tn + t_fp) if (t_tn + t_fp) > 0 else 0.0
        t_prec = precision_score(y_true, yp, zero_division=0)
        t_f1 = f1_score(y_true, yp, zero_division=0)
        sweep_results.append({
            "threshold": round(float(th), 2),
            "tp": int(t_tp), "fp": int(t_fp), "fn": int(t_fn), "tn": int(t_tn),
            "accuracy": round(float(t_acc), 4),
            "sensitivity": round(float(t_rec), 4),
            "specificity": round(float(t_spec), 4),
            "precision": round(float(t_prec), 4),
            "f1": round(float(t_f1), 4)
        })

    summary = {
        "dataset": "TopAneu-26 (Local evaluation dataset, NOT Dataset660)",
        "total_cases": len(records),
        "positive_cases": int(np.sum(y_true == 1)),
        "negative_cases": int(np.sum(y_true == 0)),
        "avg_prep_time_s": round(float(np.mean([r["preprocessing_time"] for r in records])), 4),
        "avg_infer_time_ms": round(float(np.mean([r["inference_time"] for r in records]) * 1000), 2),
        "primary_baseline_metrics": {
            "threshold": 0.5,
            "accuracy": round(float(acc), 4),
            "sensitivity_recall": round(float(rec), 4),
            "specificity": round(float(spec), 4),
            "precision": round(float(prec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "brier_score": round(float(brier), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
        },
        "modality_accuracy": round(float(mod_acc), 4),
        "threshold_sweep": sweep_results
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)
    print(f"Summary metrics saved to: {SUMMARY_JSON}")

    print("\n" + "=" * 60)
    print("P3 PRETRAINED BASELINE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Cases Evaluated: {len(records)} (Positive: {summary['positive_cases']}, Negative: {summary['negative_cases']})")
    print(f"Average Preprocessing Time: {summary['avg_prep_time_s']:.3f} s/case")
    print(f"Average GPU Inference Time: {summary['avg_infer_time_ms']:.1f} ms/case")
    print(f"Primary Metrics (Threshold = 0.5):")
    print(f"  - Accuracy:    {acc*100:.2f}%")
    print(f"  - Sensitivity: {rec*100:.2f}% (TP={tp}, FN={fn})")
    print(f"  - Specificity: {spec*100:.2f}% (TN={tn}, FP={fp})")
    print(f"  - Precision:   {prec*100:.2f}%")
    print(f"  - F1-Score:    {f1:.4f}")
    print(f"  - ROC-AUC:     {roc_auc:.4f}")
    print(f"  - PR-AUC:      {pr_auc:.4f}")
    print(f"  - Brier Score: {brier:.4f}")
    print(f"Modality Classification Accuracy: {mod_acc*100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
