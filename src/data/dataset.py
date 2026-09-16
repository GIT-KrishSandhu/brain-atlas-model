import os
import re
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


class TopAneuDataset(Dataset):
    """
    TopAneu-26 PyTorch Dataset for 3D aneurysm classification.
    
    Loads 3D volumes, resamples to (224, 224, 224) 1mm isotropic equivalent,
    applies P3 Z-score normalization, and caches preprocessed tensors for
    fast subsequent epoch iterations.
    """
    def __init__(
        self,
        split_df,
        data_dir: str = "topaneu_release",
        cache_dir: str = "scratch/cache_224",
        target_size: tuple = (224, 224, 224),
        preload_to_ram: bool = False,
        transform = None
    ):
        self.df = split_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.cache_dir = cache_dir
        self.target_size = target_size
        self.preload_to_ram = preload_to_ram
        self.transform = transform
        self.ram_cache = {}

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _preprocess_volume(self, nifti_path: str) -> np.ndarray:
        nii = nib.load(nifti_path)
        raw_vol = nii.get_fdata().astype(np.float32)

        # 3D Trilinear Resampling to target_size
        factors = [self.target_size[i] / raw_vol.shape[i] for i in range(3)]
        resampled_vol = zoom(raw_vol, factors, order=1)

        # Standard P3 Z-score normalization
        mean = resampled_vol.mean()
        std = max(resampled_vol.std(), 1e-8)
        norm_vol = (resampled_vol - mean) / std
        return norm_vol.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row["case_id"]
        presence = float(row["presence"])
        modality_str = row["modality"]
        # Modality encoding: 0=CTA, 1=MRA
        modality_idx = 1 if modality_str == "MRA" else 0

        # Check RAM cache
        if self.preload_to_ram and case_id in self.ram_cache:
            x_tensor = self.ram_cache[case_id]
        else:
            # Check Disk cache
            cache_file = os.path.join(self.cache_dir, f"{case_id}.pt") if self.cache_dir else None
            if cache_file and os.path.exists(cache_file):
                x_tensor = torch.load(cache_file, weights_only=True)
            else:
                img_path = os.path.join(self.images_dir, f"{case_id}_0000.nii.gz")
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Scan not found: {img_path}")
                norm_vol = self._preprocess_volume(img_path)
                x_tensor = torch.from_numpy(norm_vol).unsqueeze(0).float() # (1, D, H, W)
                
                if cache_file:
                    torch.save(x_tensor, cache_file)

            if self.preload_to_ram:
                self.ram_cache[case_id] = x_tensor

        # Apply stochastic on-the-fly augmentation if configured
        if self.transform is not None:
            x_tensor = self.transform(x_tensor)

        y_presence = torch.tensor([presence], dtype=torch.float32)
        y_modality = torch.tensor(modality_idx, dtype=torch.long)

        return {
            "case_id": case_id,
            "image": x_tensor,
            "presence": y_presence,
            "modality": y_modality
        }
