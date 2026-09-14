import os
import sys
import torch
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.p3 import P3Architecture


def test_p3_parameter_and_tensor_count():
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
    )

    total_params = sum(p.numel() for p in model.parameters())
    num_tensors = len(list(model.state_dict().keys()))

    assert total_params == 109_359_299, f"Parameter mismatch: {total_params}"
    assert num_tensors == 583, f"Tensor count mismatch: {num_tensors}"


def test_p3_forward_shapes():
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
    )
    model.eval()

    # Test input with 64^3 (bottleneck will be 2^3 = 8 voxels, valid for InstanceNorm3d)
    dummy_input = torch.randn(1, 1, 64, 64, 64)

    # 1. Test only_forward_cls=True
    with torch.no_grad():
        cls_preds = model(dummy_input, only_forward_cls=True)
    assert len(cls_preds) == 2
    assert cls_preds[0].shape == (1, 1)   # presence logit
    assert cls_preds[1].shape == (1, 13)  # location logits

    # 2. Test full multi-task forward
    with torch.no_grad():
        r1, r2, cls_preds, modality = model(dummy_input)
    assert len(r1) == 5
    assert len(r2) == 5
    assert cls_preds[0].shape == (1, 1)
    assert cls_preds[1].shape == (1, 13)
    assert modality.shape == (1, 4)
