import os
import sys
import glob
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def process_single_case(args):
    img_path, cache_dir, target_size = args
    fname = os.path.basename(img_path)
    case_id = fname.replace("_0000.nii.gz", "")
    out_path = os.path.join(cache_dir, f"{case_id}.pt")

    if os.path.exists(out_path):
        return case_id, "ALREADY_CACHED"

    try:
        nii = nib.load(img_path)
        raw = nii.get_fdata().astype(np.float32)
        factors = [target_size[i] / raw.shape[i] for i in range(3)]
        resampled = zoom(raw, factors, order=1)
        mean = resampled.mean()
        std = max(resampled.std(), 1e-8)
        norm = (resampled - mean) / std
        tensor = torch.from_numpy(norm.astype(np.float32)).unsqueeze(0) # (1, D, H, W)
        torch.save(tensor, out_path)
        return case_id, "CACHED"
    except Exception as e:
        return case_id, f"ERROR: {e}"


def precache_all(
    data_dir: str = "topaneu_release",
    cache_dir: str = "scratch/cache_224",
    target_size: tuple = (224, 224, 224),
    max_workers: int = 16
):
    print("=" * 70)
    print("PRE-CACHING TOPANEU-26 224^3 ISOTROPIC RESAMPLED VOLUMES")
    print("=" * 70)
    print(f"Data Dir: {data_dir}")
    print(f"Cache Dir: {cache_dir}")
    print(f"Workers: {max_workers}")

    os.makedirs(cache_dir, exist_ok=True)
    images_dir = os.path.join(data_dir, "images")
    img_files = sorted(glob.glob(os.path.join(images_dir, "*_0000.nii.gz")))
    img_files = [f for f in img_files if not os.path.basename(f).startswith("._")]
    print(f"Total scans to verify/cache: {len(img_files)}")

    tasks = [(f, cache_dir, target_size) for f in img_files]
    t0 = time.time()
    cached_count = 0
    already_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, (cid, status) in enumerate(executor.map(process_single_case, tasks), 1):
            if status == "CACHED":
                cached_count += 1
            elif status == "ALREADY_CACHED":
                already_count += 1
            else:
                error_count += 1
                print(f"Error on {cid}: {status}")

            if idx % 50 == 0 or idx == len(tasks):
                print(f"  Processed {idx:>3d}/{len(tasks)} | New Cached: {cached_count} | Existing: {already_count}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.2f}s!")
    print(f"New Cached: {cached_count}, Already Cached: {already_count}, Errors: {error_count}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="topaneu_release")
    parser.add_argument("--cache_dir", type=str, default="scratch/cache_224")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    precache_all(args.data_dir, args.cache_dir, max_workers=args.workers)
