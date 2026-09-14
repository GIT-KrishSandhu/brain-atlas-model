import os
import sys
import argparse
import yaml
import torch
import pandas as pd

# Add repository root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.p3 import P3Architecture
from src.data.dataset import TopAneuDataset
from src.training.trainer import P3Trainer


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train / Fine-tune P3 on TopAneu-26")
    parser.add_argument("--config", type=str, default="configs/p4_p3_finetune.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.lr is not None:
        config["lr"] = args.lr

    # Set deterministic seeds
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=" * 70)
    print("PHASE 4: INDEPENDENT P3 FINE-TUNING PIPELINE")
    print("=" * 70)
    print(f"Config File: {args.config}")
    print(f"Experiment ID: {config.get('experiment_id')}")

    # 1. Load Dataset Splits
    split_csv = config["split_csv"]
    df_splits = pd.read_csv(split_csv)
    train_df = df_splits[df_splits["split"] == "train"].reset_index(drop=True)
    val_df = df_splits[df_splits["split"] == "val"].reset_index(drop=True)

    print(f"Loaded Splits: Train={len(train_df)} cases, Val={len(val_df)} cases")

    train_dataset = TopAneuDataset(
        split_df=train_df,
        data_dir=config.get("data_dir", "topaneu_release"),
        cache_dir=config.get("cache_dir", "scratch/cache_224"),
        target_size=tuple(config.get("target_size", [224, 224, 224]))
    )
    val_dataset = TopAneuDataset(
        split_df=val_df,
        data_dir=config.get("data_dir", "topaneu_release"),
        cache_dir=config.get("cache_dir", "scratch/cache_224"),
        target_size=tuple(config.get("target_size", [224, 224, 224]))
    )

    # 2. Build Independent Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Constructing P3Architecture on {device}...")
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

    # 3. Load Official Pretrained Checkpoint
    ckpt_path = config["checkpoint_path"]
    print(f"Loading official pretrained weights: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_res = model.load_state_dict(ckpt["network_weights"], strict=True)
    print(f"Strict loading: {load_res}")

    # 4. Train
    trainer = P3Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device
    )

    history = trainer.train()
    print("Training finished successfully.")


if __name__ == "__main__":
    main()
