import os
import sys
import argparse
import json
import torch
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.p3 import P3Architecture
from src.data.dataset import TopAneuDataset
from src.evaluation.evaluator import P3Evaluator


def evaluate(
    checkpoint_path: str,
    split_csv: str = "experiments/splits/topaneu_v1.csv",
    split_name: str = "test",
    output_dir: str = "experiments/results",
    state_name: str = "evaluation"
):
    print("=" * 70)
    print(f"P3 EVALUATION PIPELINE: [{state_name}] on [{split_name.upper()}] SPLIT")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split CSV: {split_csv}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Dataset
    df_splits = pd.read_csv(split_csv)
    if split_name != "all":
        df_eval = df_splits[df_splits["split"] == split_name].reset_index(drop=True)
    else:
        df_eval = df_splits.reset_index(drop=True)

    print(f"Total Cases to Evaluate: {len(df_eval)} (Pos: {df_eval['presence'].sum()}, Neg: {(df_eval['presence']==0).sum()})")

    eval_dataset = TopAneuDataset(
        split_df=df_eval,
        data_dir="topaneu_release",
        cache_dir="scratch/cache_224",
        target_size=(224, 224, 224)
    )

    # 2. Build Model
    model = P3Architecture(
        in_channels=1,
        out_channels_1=15,
        out_channels_2=14,
        cls_head_num_classes_list=[1, 13],
        cls_drop_out_list=[0.0, 0.0],
        cls_query_num_list=[2, 16],
        use_cross_attention=True,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        kernel_sizes=[[3, 3, 3]] * 6,
        strides=[[1, 1, 1]] + [[2, 2, 2]] * 5,
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=True
    ).to(device)

    # 3. Load Checkpoint (support official raw dict or training saved dict)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "network_weights" in ckpt:
        state_dict = ckpt["network_weights"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    print("Checkpoint loaded successfully with strict=True.")

    # 4. Evaluate
    evaluator = P3Evaluator(model=model, device=device)
    metrics, df_preds = evaluator.evaluate_dataset(eval_dataset)

    # Print summary
    print("\n--- Evaluation Results ---")
    print(f"Accuracy   : {metrics['accuracy']*100:.2f}%")
    print(f"Sensitivity: {metrics['sensitivity']*100:.2f}% (Recall)")
    print(f"Specificity: {metrics['specificity']*100:.2f}%")
    print(f"Precision  : {metrics['precision']*100:.2f}% (PPV)")
    print(f"F1-Score   : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC     : {metrics['pr_auc']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
    cm = metrics['confusion_matrix']
    print(f"Confusion Matrix: TP={cm['tp']}, FP={cm['fp']}, TN={cm['tn']}, FN={cm['fn']}")
    print(f"Mean Latency: {metrics['mean_inference_time_ms']:.1f} ms/case")

    # 5. Save Artifacts
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f"{state_name}_{split_name}_metrics.json")
    preds_path = os.path.join(output_dir, f"{state_name}_{split_name}_predictions.csv")

    with open(metrics_path, "w") as jf:
        json.dump(metrics, jf, indent=2)
    df_preds.to_csv(preds_path, index=False)

    print(f"\nArtifacts saved:")
    print(f"  Metrics JSON: {metrics_path}")
    print(f"  Predictions: {preds_path}")
    print("=" * 70)

    return metrics, df_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split_csv", type=str, default="experiments/splits/topaneu_v1.csv")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    parser.add_argument("--state_name", type=str, default="eval")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        split_csv=args.split_csv,
        split_name=args.split,
        output_dir=args.output_dir,
        state_name=args.state_name
    )
