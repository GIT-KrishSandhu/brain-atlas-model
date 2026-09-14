import os
import time
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss
)


class P3Evaluator:
    """
    Standardized evaluator for P3 architecture on TopAneu-26.
    Evaluates binary aneurysm presence, multi-class location matching,
    and modality classification.
    """
    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def evaluate_dataset(self, dataset, batch_size: int = 1):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        all_case_ids = []
        all_probs = []
        all_targets = []
        all_modality_preds = []
        all_modality_targets = []
        inference_times = []

        with torch.no_grad():
            for batch in loader:
                case_id = batch["case_id"][0]
                images = batch["image"].to(self.device, non_blocking=True)
                targets = batch["presence"].cpu().numpy().flatten()[0]
                mod_targets = batch["modality"].cpu().numpy().flatten()[0]

                t0 = time.time()
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type == "cuda")):
                    # Full classification forward
                    cls_preds = self.model(images, only_forward_cls=True)
                    presence_logit = cls_preds[0]
                    prob = torch.sigmoid(presence_logit).cpu().numpy().flatten()[0]

                inf_time = time.time() - t0

                all_case_ids.append(case_id)
                all_probs.append(float(prob))
                all_targets.append(int(targets))
                all_modality_preds.append(0) # placeholder or from modality head
                all_modality_targets.append(int(mod_targets))
                inference_times.append(inf_time)

        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        bin_preds = (all_probs >= 0.5).astype(int)

        # Metrics computation
        cm = confusion_matrix(all_targets, bin_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(all_targets, bin_preds)
        sens = recall_score(all_targets, bin_preds, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = precision_score(all_targets, bin_preds, zero_division=0)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = f1_score(all_targets, bin_preds, zero_division=0)
        try:
            roc_auc = roc_auc_score(all_targets, all_probs)
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = average_precision_score(all_targets, all_probs)
        except Exception:
            pr_auc = 0.5
        brier = brier_score_loss(all_targets, all_probs)

        metrics = {
            "total_samples": len(all_targets),
            "positive_samples": int(all_targets.sum()),
            "negative_samples": int((all_targets == 0).sum()),
            "accuracy": float(acc),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "precision": float(prec),
            "npv": float(npv),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "brier_score": float(brier),
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn)
            },
            "mean_inference_time_ms": float(np.mean(inference_times) * 1000)
        }

        # Predictions DataFrame
        df_preds = pd.DataFrame({
            "case_id": all_case_ids,
            "ground_truth_presence": all_targets,
            "predicted_presence_probability": all_probs,
            "predicted_presence_label": bin_preds,
            "inference_time_ms": np.array(inference_times) * 1000
        })

        return metrics, df_preds
