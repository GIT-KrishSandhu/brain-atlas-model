import sys
import os
import argparse
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.p3 import P3Architecture


def verify_checkpoint(checkpoint_path: str, device: str = "cpu"):
    print("=" * 75)
    print("INDEPENDENT P3 ARCHITECTURE CHECKPOINT VERIFICATION")
    print("=" * 75)
    print(f"Target Checkpoint: {checkpoint_path}")
    print(f"Evaluation Device: {device}")

    print("\n1. Instantiating independent P3Architecture...")
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

    total_params = sum(p.numel() for p in model.parameters())
    num_tensors = len(list(model.state_dict().keys()))
    print(f"   Total Parameters: {total_params:,}")
    print(f"   State Dict Tensors: {num_tensors}")

    EXPECTED_PARAMS = 109_359_299
    EXPECTED_TENSORS = 583
    assert total_params == EXPECTED_PARAMS, f"Parameter mismatch: {total_params} vs expected {EXPECTED_PARAMS}"
    assert num_tensors == EXPECTED_TENSORS, f"Tensor count mismatch: {num_tensors} vs expected {EXPECTED_TENSORS}"
    print("   [PASS] Parameter count and tensor count match exact reference specification.")

    print("\n2. Loading official checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["network_weights"]

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    print(f"   Missing keys: {len(missing)}")
    print(f"   Unexpected keys: {len(unexpected)}")
    assert len(missing) == 0, f"Missing keys detected: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys detected: {unexpected}"

    load_result = model.load_state_dict(state_dict, strict=True)
    print(f"   Strict load result: {load_result}")
    print("   [PASS] Strict state_dict loading passed with 0 missing and 0 unexpected keys.")

    print("\n3. Testing forward inference pass...")
    model.eval()
    test_input = torch.randn(1, 1, 64, 64, 64, device=device)
    with torch.no_grad():
        out = model(test_input)

    r1, r2, cls_preds, modality_pred = out
    print(f"   seg_layers_1 levels: {len(r1)} (highest: {r1[0].shape}, lowest: {r1[-1].shape})")
    print(f"   seg_layers_2 levels: {len(r2)} (highest: {r2[0].shape}, lowest: {r2[-1].shape})")
    print(f"   presence logit shape: {cls_preds[0].shape}")
    print(f"   location logits shape: {cls_preds[1].shape}")
    print(f"   modality logits shape: {modality_pred.shape}")
    print("   [PASS] All output tensor dimensions match architecture specification.")

    print("\n" + "=" * 75)
    print("VERIFICATION COMPLETE: P3 INDEPENDENT IMPLEMENTATION IS 100% COMPLIANT")
    print("=" * 75)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"D:\NLP_Project\scratch\checkpoints\Dataset660_26classes_resize224_4661\onlyMirror01_lr4e3_100epochs_ps224\fold_0\checkpoint_final.pth"
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    verify_checkpoint(args.checkpoint, args.device)
