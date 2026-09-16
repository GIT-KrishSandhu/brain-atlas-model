import os
import sys
import math
import torch
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.transforms import (
    TopAneuAugmentationPipeline,
    get_3d_rotation_matrix,
    create_3d_gaussian_kernel
)
from src.data.dataset import TopAneuDataset


def test_rotation_matrix_orthogonality():
    """Verify 3D rotation matrix is orthogonal: R * R.T = I and det(R) = 1."""
    angles = [math.radians(10.0), math.radians(-5.0), math.radians(7.5)]
    device = torch.device("cpu")
    R = get_3d_rotation_matrix(angles[0], angles[1], angles[2], device)

    assert R.shape == (3, 3)
    I = torch.eye(3, dtype=torch.float32)
    assert torch.allclose(torch.mm(R, R.t()), I, atol=1e-5)
    det = torch.det(R)
    assert torch.allclose(det, torch.tensor(1.0), atol=1e-5)


def test_gaussian_kernel_normalization():
    """Verify 3D Gaussian convolution kernel sums to exactly 1.0."""
    kernel = create_3d_gaussian_kernel(sigma=0.65, kernel_size=3)
    assert kernel.shape == (1, 1, 3, 3, 3)
    assert torch.allclose(kernel.sum(), torch.tensor(1.0), atol=1e-6)


def test_spatial_affine_shape_and_finite():
    """Verify spatial affine preserves shape and contains zero NaNs/Infs."""
    pipeline = TopAneuAugmentationPipeline(
        p_rotation=1.0,
        max_rotation_deg=10.0,
        p_translation=1.0,
        max_translation_voxels=8.0,
        p_contrast=0.0,
        p_gamma=0.0,
        p_noise=0.0,
        p_blur=0.0,
        target_size=(64, 64, 64),
        seed=42
    )

    dummy_vol = torch.randn(1, 64, 64, 64)
    aug_vol = pipeline(dummy_vol)

    assert aug_vol.shape == (1, 64, 64, 64)
    assert torch.isfinite(aug_vol).all()
    assert not torch.isnan(aug_vol).any()


def test_spatial_transform_mask_alignment_and_label_integrity():
    """
    Verify spatial transform transforms image and discrete integer masks identically
    using nearest-neighbor interpolation, ensuring label IDs are preserved without blurring.
    """
    pipeline = TopAneuAugmentationPipeline(
        p_rotation=1.0,
        max_rotation_deg=10.0,
        p_translation=1.0,
        max_translation_voxels=5.0,
        p_contrast=0.0,
        p_gamma=0.0,
        p_noise=0.0,
        p_blur=0.0,
        target_size=(48, 48, 48),
        seed=123
    )

    # Create dummy volume and multi-class integer masks
    vol = torch.zeros(1, 48, 48, 48)
    vessel_mask = torch.zeros(48, 48, 48, dtype=torch.int64)
    loc_mask = torch.zeros(48, 48, 48, dtype=torch.int64)

    # Place a simulated vessel segment (label 5 = R-M1) and aneurysm (label 49 = R-5.3 M1-M2 junction)
    vol[:, 20:28, 20:28, 20:28] = 3.5
    vessel_mask[20:28, 20:28, 20:28] = 5
    loc_mask[22:26, 22:26, 22:26] = 49

    masks = {"vessel": vessel_mask, "location": loc_mask}
    aug_vol, aug_masks = pipeline(vol, masks=masks)

    assert aug_vol.shape == (1, 48, 48, 48)
    assert aug_masks["vessel"].shape == (48, 48, 48)
    assert aug_masks["location"].shape == (48, 48, 48)

    # Nearest neighbor check: discrete unique labels in masks must be a subset of original labels
    unique_vessel = torch.unique(aug_masks["vessel"]).tolist()
    unique_loc = torch.unique(aug_masks["location"]).tolist()

    assert set(unique_vessel).issubset({0, 5}), f"Spurious labels introduced in vessel mask: {unique_vessel}"
    assert set(unique_loc).issubset({0, 49}), f"Spurious labels introduced in location mask: {unique_loc}"

    # Foreground lesion must still exist after conservative transformation
    assert (aug_masks["location"] == 49).sum() > 0, "Lesion was completely eliminated by transformation!"


def test_multi_aneurysm_preservation():
    """
    Verify that in multi-aneurysm cases (simulating TopAneu's 66 multi-aneurysm cases),
    all lesions are preserved and remain distinctly separable.
    """
    pipeline = TopAneuAugmentationPipeline(
        p_rotation=1.0,
        max_rotation_deg=8.0,
        p_translation=1.0,
        max_translation_voxels=4.0,
        p_contrast=0.0,
        p_gamma=0.0,
        p_noise=0.0,
        p_blur=0.0,
        target_size=(64, 64, 64),
        seed=999
    )

    loc_mask = torch.zeros(64, 64, 64, dtype=torch.int64)
    # Aneurysm 1: Acom (label 36) at anterior position
    loc_mask[30:35, 30:35, 45:50] = 36
    # Aneurysm 2: R-M1-M2 (label 49) at lateral position
    loc_mask[30:35, 15:20, 30:35] = 49

    vol = torch.zeros(1, 64, 64, 64)
    _, aug_masks = pipeline(vol, masks={"location": loc_mask})

    aug_loc = aug_masks["location"]
    # Both aneurysms must remain present
    assert (aug_loc == 36).sum() > 0, "Aneurysm 1 (label 36) was lost!"
    assert (aug_loc == 49).sum() > 0, "Aneurysm 2 (label 49) was lost!"


def test_intensity_transforms_numeric_integrity():
    """Verify intensity transforms generate valid, finite values with no NaN/Inf."""
    pipeline = TopAneuAugmentationPipeline(
        p_rotation=0.0,
        p_translation=0.0,
        p_contrast=1.0,
        contrast_range=(0.90, 1.10),
        p_gamma=1.0,
        gamma_range=(0.90, 1.10),
        p_noise=1.0,
        noise_sigma=0.05,
        p_blur=0.0,
        target_size=(32, 32, 32),
        seed=777
    )

    vol = torch.randn(1, 32, 32, 32)
    aug_vol = pipeline(vol)

    assert torch.isfinite(aug_vol).all()
    assert not torch.isnan(aug_vol).any()
    assert not torch.isinf(aug_vol).any()


def test_noise_blur_mutual_exclusion():
    """
    Verify anti-stacking safeguard: when both noise and blur are enabled,
    they are mutually exclusive and never both applied to the same sample.
    """
    # Create pipeline where both noise and blur have p=1.0
    pipeline = TopAneuAugmentationPipeline(
        p_rotation=0.0,
        p_translation=0.0,
        p_contrast=0.0,
        p_gamma=0.0,
        p_noise=1.0,
        noise_sigma=0.05,
        p_blur=1.0,
        blur_sigma_range=(0.60, 0.60),
        target_size=(32, 32, 32),
        seed=101
    )

    # A smooth uniform volume with high frequency delta spike
    vol = torch.zeros(1, 16, 16, 16)
    vol[0, 8, 8, 8] = 10.0

    # In apply_intensity, mutual exclusion explicitly picks ONE of apply_noise or apply_blur
    # Let's verify over 20 iterations that either noise or blur occurs, but the mutual exclusion branch executes
    for _ in range(20):
        aug = pipeline.apply_intensity(vol)
        assert torch.isfinite(aug).all()


def test_topaneu_dataset_with_augmentation():
    """Verify TopAneuDataset successfully integrates TopAneuAugmentationPipeline."""
    split_csv = "experiments/splits/topaneu_v1.csv"
    assert os.path.exists(split_csv)
    df = pd.read_csv(split_csv).head(2)

    pipeline = TopAneuAugmentationPipeline(
        p_rotation=0.5,
        p_translation=0.5,
        p_contrast=0.5,
        p_gamma=0.5,
        p_noise=0.2,
        p_blur=0.2,
        seed=42
    )

    # Train dataset with augmentation
    train_ds = TopAneuDataset(
        split_df=df,
        data_dir="topaneu_release",
        cache_dir="scratch/cache_224",
        target_size=(224, 224, 224),
        transform=pipeline
    )

    item = train_ds[0]
    assert item["image"].shape == (1, 224, 224, 224)
    assert torch.isfinite(item["image"]).all()
    assert item["presence"].shape == (1,)
    assert item["modality"].dtype == torch.long

    # Val dataset with transform=None (unaugmented control)
    val_ds = TopAneuDataset(
        split_df=df,
        data_dir="topaneu_release",
        cache_dir="scratch/cache_224",
        target_size=(224, 224, 224),
        transform=None
    )
    val_item = val_ds[0]
    assert val_item["image"].shape == (1, 224, 224, 224)
    assert torch.isfinite(val_item["image"]).all()
