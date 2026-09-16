import math
import random
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union


def get_3d_rotation_matrix(rx: float, ry: float, rz: float, device: torch.device) -> torch.Tensor:
    """
    Construct 3D rotation matrix R = Rz * Ry * Rx for angles in radians.
    Coordinates are (x, y, z) mapped to (W, H, D).
    """
    cos_x, sin_x = math.cos(rx), math.sin(rx)
    cos_y, sin_y = math.cos(ry), math.sin(ry)
    cos_z, sin_z = math.cos(rz), math.sin(rz)

    Rx = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, cos_x, -sin_x],
        [0.0, sin_x, cos_x]
    ], dtype=torch.float32, device=device)

    Ry = torch.tensor([
        [cos_y, 0.0, sin_y],
        [0.0, 1.0, 0.0],
        [-sin_y, 0.0, cos_y]
    ], dtype=torch.float32, device=device)

    Rz = torch.tensor([
        [cos_z, -sin_z, 0.0],
        [sin_z, cos_z, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device)

    return torch.mm(Rz, torch.mm(Ry, Rx))


def create_3d_gaussian_kernel(sigma: float, kernel_size: int = 3, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Creates a 3D Gaussian convolution kernel of shape (1, 1, K, K, K) normalized to sum=1.
    """
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k1d = torch.exp(-coords**2 / (2.0 * sigma**2))
    k1d = k1d / k1d.sum()

    k3d = k1d[:, None, None] * k1d[None, :, None] * k1d[None, None, :]
    k3d = k3d / k3d.sum()
    return k3d.unsqueeze(0).unsqueeze(0) # (1, 1, K, K, K)


class TopAneuAugmentationPipeline:
    """
    Controlled, reproducible data augmentation pipeline for 3D neuroimaging (TopAneu-26).
    Applies approved spatial transformations (small 3D rotation, small 3D translation)
    using single-pass affine grid sampling, and mild intensity transformations
    (contrast, gamma, noise, blur) with mutual exclusion safeguards.
    """
    def __init__(
        self,
        p_rotation: float = 0.40,
        max_rotation_deg: float = 10.0,
        p_translation: float = 0.40,
        max_translation_voxels: float = 8.0,
        p_contrast: float = 0.30,
        contrast_range: Tuple[float, float] = (0.90, 1.10),
        p_gamma: float = 0.30,
        gamma_range: Tuple[float, float] = (0.90, 1.10),
        p_noise: float = 0.20,
        noise_sigma: float = 0.05,
        p_blur: float = 0.20,
        blur_sigma_range: Tuple[float, float] = (0.50, 0.75),
        target_size: Tuple[int, int, int] = (224, 224, 224),
        seed: Optional[int] = None
    ):
        self.p_rotation = p_rotation
        self.max_rotation_deg = max_rotation_deg
        self.p_translation = p_translation
        self.max_translation_voxels = max_translation_voxels
        self.p_contrast = p_contrast
        self.contrast_range = contrast_range
        self.p_gamma = p_gamma
        self.gamma_range = gamma_range
        self.p_noise = p_noise
        self.noise_sigma = noise_sigma
        self.p_blur = p_blur
        self.blur_sigma_range = blur_sigma_range
        self.target_size = target_size

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def _sample_spatial_affine(self, device: torch.device, spatial_shape: Tuple[int, int, int]) -> Tuple[bool, torch.Tensor]:
        """
        Samples a single 3x4 affine matrix combining 3D rotation and translation.
        Returns (has_transform, theta_matrix).
        """
        apply_rot = random.random() < self.p_rotation
        apply_trans = random.random() < self.p_translation

        if not apply_rot and not apply_trans:
            return False, torch.empty(0)

        # 1. Rotation
        if apply_rot:
            deg_x = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
            deg_y = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
            deg_z = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
            rad_x = math.radians(deg_x)
            rad_y = math.radians(deg_y)
            rad_z = math.radians(deg_z)
            R = get_3d_rotation_matrix(rad_x, rad_y, rad_z, device)
        else:
            R = torch.eye(3, dtype=torch.float32, device=device)

        # 2. Translation (in normalized grid coordinates [-1, 1])
        # In PyTorch 3D affine_grid, coord order is (x=W, y=H, z=D)
        D, H, W = spatial_shape
        if apply_trans:
            shift_x = random.uniform(-self.max_translation_voxels, self.max_translation_voxels)
            shift_y = random.uniform(-self.max_translation_voxels, self.max_translation_voxels)
            shift_z = random.uniform(-self.max_translation_voxels, self.max_translation_voxels)
            # Normalized shift: 2.0 * voxels / size
            norm_tx = 2.0 * shift_x / max(W, 1)
            norm_ty = 2.0 * shift_y / max(H, 1)
            norm_tz = 2.0 * shift_z / max(D, 1)
            t = torch.tensor([norm_tx, norm_ty, norm_tz], dtype=torch.float32, device=device).unsqueeze(1)
        else:
            t = torch.zeros((3, 1), dtype=torch.float32, device=device)

        # PyTorch affine_grid maps output coordinates (x_out) to input coordinates (x_in):
        # x_in = R_inv * x_out - R_inv * t = R.T * x_out - R.T * t
        R_inv = R.t()
        t_inv = -torch.mm(R_inv, t)

        theta = torch.cat([R_inv, t_inv], dim=1).unsqueeze(0) # (1, 3, 4)
        return True, theta

    def apply_spatial(
        self,
        image: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Applies a single sampled 3D affine transformation to image and optional masks.
        """
        spatial_shape = tuple(image.shape[-3:])
        has_transform, theta = self._sample_spatial_affine(image.device, spatial_shape)
        if not has_transform:
            return image, masks

        orig_ndim = image.ndim
        if orig_ndim == 3:
            img_5d = image.unsqueeze(0).unsqueeze(0)
        elif orig_ndim == 4:
            img_5d = image.unsqueeze(0)
        else:
            img_5d = image

        grid = F.affine_grid(theta, img_5d.shape, align_corners=False)

        # Trilinear interpolation with border padding mode for image
        aug_img = F.grid_sample(
            img_5d,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        )
        if orig_ndim == 3:
            aug_img = aug_img.squeeze(0).squeeze(0)
        elif orig_ndim == 4:
            aug_img = aug_img.squeeze(0)

        aug_masks = None
        if masks is not None:
            aug_masks = {}
            for name, mask in masks.items():
                m_ndim = mask.ndim
                if m_ndim == 3:
                    mask_5d = mask.unsqueeze(0).unsqueeze(0).float()
                elif m_ndim == 4:
                    mask_5d = mask.unsqueeze(0).float()
                else:
                    mask_5d = mask.float()

                # Nearest-neighbor interpolation for discrete masks to preserve class labels
                m_aug = F.grid_sample(
                    mask_5d,
                    grid,
                    mode="nearest",
                    padding_mode="border",
                    align_corners=False
                )
                if m_ndim == 3:
                    m_aug = m_aug.squeeze(0).squeeze(0)
                elif m_ndim == 4:
                    m_aug = m_aug.squeeze(0)
                aug_masks[name] = m_aug.to(mask.dtype)

        return aug_img, aug_masks

    def apply_intensity(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies mild intensity transformations in sequence.

        EPISTEMIC NOTES:
          - Contrast variation (I * c): Post-normalization multiplicative perturbation of
            zero-mean Z-score data. This is NOT raw scanner intensity scaling, HU calibration,
            or modality-specific gain modulation. [INFERRED]
          - Gamma transformation: Applied via dynamic per-volume [0,1] min-max rescaling
            then power-law then inverse rescaling. This is NOT physical scanner gamma correction
            or a measured transfer-function model. It is a relative contrast reshape of the
            normalized representation. [INFERRED]
          - Gaussian noise sigma=0.05 is an experimental hyperparameter, NOT a measured
            physical noise level from any specific acquisition. [INFERRED]

        Anti-stacking rules:
          - Noise and blur are mutually exclusive per sample presentation.
          - NaN/Inf on output falls back to the original image.
        """
        vol = image.clone()

        # 1. Post-normalization contrast perturbation: I_aug = I * c, c ~ U(0.90, 1.10)
        #    Because I is zero-mean Z-score, this rescales variance around the mean.
        #    This is NOT raw scanner intensity scaling. [INFERRED]
        if random.random() < self.p_contrast:
            c = random.uniform(self.contrast_range[0], self.contrast_range[1])
            vol = vol * c

        # 2. Relative contrast reshape via dynamic [0,1] rescaling and power-law.
        #    Operates on the normalized representation, NOT on raw physical intensities.
        #    gamma in [0.90, 1.10] produces a monotonic, high-correlation transformation. [INFERRED]
        if random.random() < self.p_gamma:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            v_min = vol.min()
            v_max = vol.max()
            v_range = v_max - v_min
            if v_range > 1e-6:
                v_scaled = (vol - v_min) / v_range          # [0, 1]
                v_gamma = torch.clamp(v_scaled, 0.0, 1.0) ** gamma
                vol = v_gamma * v_range + v_min              # inverse rescale

        # 3. Noise XOR blur mutual exclusion
        apply_noise = random.random() < self.p_noise
        apply_blur = random.random() < self.p_blur

        if apply_noise and apply_blur:
            # Enforce mutual exclusion: never both on the same sample
            if random.random() < 0.5:
                apply_blur = False
            else:
                apply_noise = False

        if apply_noise:
            # Additive Gaussian noise: experimental hyperparameter, NOT measured noise level [INFERRED]
            noise = torch.randn_like(vol) * self.noise_sigma
            vol = vol + noise

        elif apply_blur:
            sigma = random.uniform(self.blur_sigma_range[0], self.blur_sigma_range[1])
            kernel = create_3d_gaussian_kernel(sigma=sigma, kernel_size=3, device=vol.device)
            # Ensure input is exactly (N, C, D, H, W) for conv3d
            orig_ndim = vol.ndim
            if orig_ndim == 3:          # (D, H, W) -> (1, 1, D, H, W)
                vol_5d = vol.unsqueeze(0).unsqueeze(0)
            elif orig_ndim == 4:        # (C, D, H, W) -> (1, C, D, H, W)
                vol_5d = vol.unsqueeze(0)
            else:
                vol_5d = vol            # already (N, C, D, H, W)
            blurred = F.conv3d(vol_5d, kernel, padding=1)
            if orig_ndim == 3:
                vol = blurred.squeeze(0).squeeze(0)
            elif orig_ndim == 4:
                vol = blurred.squeeze(0)
            else:
                vol = blurred

        # Numeric integrity: fall back to original on any non-finite value
        if not torch.isfinite(vol).all():
            return image

        return vol

    def __call__(
        self,
        image: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Full stochastic forward pass: spatial affine followed by intensity transformation.
        """
        # 1. Spatial affine transform (applies identically to image and masks)
        aug_img, aug_masks = self.apply_spatial(image, masks)

        # 2. Intensity transform (applies only to image, never to masks)
        aug_img = self.apply_intensity(aug_img)

        if masks is not None:
            return aug_img, aug_masks
        return aug_img
