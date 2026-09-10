"""
STEP 5 — Visualizations: actual-data slices, mask overlays.
Saves PNG files to reports/data_survey/visualizations/
READ-ONLY on original data.
"""
import pathlib, json, sys
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT     = pathlib.Path(r"d:\NLP_Project\topaneu_release")
OUT_VIZ  = pathlib.Path(r"d:\NLP_Project\reports\data_survey\visualizations")
OUT_FIG  = pathlib.Path(r"d:\NLP_Project\reports\data_survey\figures")
OUT_VIZ.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

IMG_DIR = ROOT / "images"
VES_DIR = ROOT / "vessel_masks"
LOC_DIR = ROOT / "location_masks"
TYP_DIR = ROOT / "type_masks"
JSN_DIR = ROOT / "location_jsons"

vessel_map   = {v:k for k,v in json.load(open(ROOT/"vessel_mapping.json"))["labels"].items()}
location_map = {v:k for k,v in json.load(open(ROOT/"location_mapping.json"))["labels"].items()}
type_map     = {v:k for k,v in json.load(open(ROOT/"type_mapping.json"))["labels"].items()}

def normalize(vol, p1=1, p99=99):
    lo, hi = np.percentile(vol, p1), np.percentile(vol, p99)
    return np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1)

def mid_slices(vol):
    """Return axial, coronal, sagittal mid-slices."""
    sx, sy, sz = vol.shape[:3]
    return vol[sx//2, :, :], vol[:, sy//2, :], vol[:, :, sz//2]

def save_orthogonal(case_id, img_data, mask_data=None, mask_cmap=None, suffix="", alpha=0.4, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    planes = [("Sagittal", img_data[img_data.shape[0]//2, :, :]),
              ("Coronal",  img_data[:, img_data.shape[1]//2, :]),
              ("Axial",    img_data[:, :, img_data.shape[2]//2])]
    for ax, (plane_name, sl) in zip(axes, planes):
        sl_norm = normalize(sl)
        ax.imshow(sl_norm.T, cmap="gray", origin="lower", aspect="auto")
        if mask_data is not None:
            msl = [mask_data[mask_data.shape[0]//2, :, :],
                   mask_data[:, mask_data.shape[1]//2, :],
                   mask_data[:, :, mask_data.shape[2]//2]][["Sagittal","Coronal","Axial"].index(plane_name)]
            masked = np.ma.masked_where(msl == 0, msl)
            ax.imshow(masked.T, cmap=mask_cmap or "jet", alpha=alpha, origin="lower", aspect="auto",
                      vmin=1, vmax=max(mask_data.max(), 1))
        ax.set_title(f"{plane_name}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{case_id} — {title}", fontsize=11)
    plt.tight_layout()
    out = OUT_VIZ / f"{case_id}_{suffix}.png"
    plt.savefig(str(out), dpi=100, bbox_inches="tight")
    plt.close()
    return out

# Pick representative cases: some positive, some negative
all_jsons = sorted([f for f in JSN_DIR.iterdir() if not f.name.startswith("._")])
positive = []
negative = []
for jp in all_jsons:
    with open(jp) as f:
        d = json.load(f)
    cid = jp.stem
    img_fp = IMG_DIR / f"{cid}_0000.nii.gz"
    if not img_fp.exists():
        continue
    if d.get("locations"):
        positive.append((cid, d["locations"]))
    else:
        negative.append(cid)

print(f"Positive: {len(positive)}, Negative: {len(negative)}")

# Select cases: first 3 positive (with single aneurysm), first 2 negative, 1 multi-aneurysm
single_pos = [(c,l) for c,l in positive if len(l)==1][:3]
multi_pos  = [(c,l) for c,l in positive if len(l)>1][:1]
neg_sel    = negative[:2]
selected   = [(c, "positive_single", l) for c,l in single_pos] + \
             [(c, "positive_multi",  l) for c,l in multi_pos]  + \
             [(c, "negative",        []) for c in neg_sel]

print(f"\nGenerating visualizations for {len(selected)} cases:")
for case_id, kind, locs in selected:
    print(f"  {case_id} ({kind}) locs={locs}")
    img_fp = IMG_DIR / f"{case_id}_0000.nii.gz"
    ves_fp = VES_DIR / f"{case_id}.nii.gz"
    loc_fp = LOC_DIR / f"{case_id}.nii.gz"
    typ_fp = TYP_DIR / f"{case_id}.nii.gz"

    try:
        img_data = nib.load(str(img_fp)).get_fdata(dtype=np.float32)
        # Raw image
        save_orthogonal(case_id, img_data, suffix="image", title="Raw Image")

        # Vessel mask overlay
        if ves_fp.exists():
            ves_data = nib.load(str(ves_fp)).get_fdata(dtype=np.float32)
            save_orthogonal(case_id, img_data, ves_data, suffix="vessel_overlay", title="+ Vessel Mask")

        # Location mask overlay
        if loc_fp.exists():
            loc_data = nib.load(str(loc_fp)).get_fdata(dtype=np.float32)
            save_orthogonal(case_id, img_data, loc_data, mask_cmap="nipy_spectral",
                            suffix="location_overlay", title="+ Location Mask")

        # Type mask overlay
        if typ_fp.exists():
            typ_data = nib.load(str(typ_fp)).get_fdata(dtype=np.float32)
            save_orthogonal(case_id, img_data, typ_data, mask_cmap="Reds",
                            suffix="type_overlay", title="+ Type Mask")
        print(f"    -> saved 4 PNGs")
    except Exception as e:
        print(f"    ERROR: {e}")

print("\nVisualizations done.")

# ── Distribution figures ────────────────────────────────────────────────────────
import csv as csvmod

imaging_csv = pathlib.Path(r"d:\NLP_Project\reports\data_survey\imaging_characterization.csv")
if imaging_csv.exists():
    with open(imaging_csv, encoding="utf-8") as f:
        rdr = csvmod.DictReader(f)
        img_rows = list(rdr)

    # Voxel spacing distribution
    vx = [float(r["voxel_x_mm"]) for r in img_rows]
    vy = [float(r["voxel_y_mm"]) for r in img_rows]
    vz = [float(r["voxel_z_mm"]) for r in img_rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, vals, label in zip(axes, [vx,vy,vz], ["X (mm)","Y (mm)","Z (mm)"]):
        ax.hist(vals, bins=30, color="#3B82F6", edgecolor="white", linewidth=0.5)
        ax.set_xlabel(label); ax.set_ylabel("Count")
        ax.set_title(f"Voxel Spacing — {label}")
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("TopAneu-26: Voxel Spacing Distribution (n=415 volumes)", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(OUT_FIG/"voxel_spacing_distribution.png"), dpi=120, bbox_inches="tight")
    plt.close(); print("Saved voxel_spacing_distribution.png")

    # Slice count distribution
    sz_vals = [int(r["shape_z"]) for r in img_rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sz_vals, bins=30, color="#8B5CF6", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of slices (Z-dim)"); ax.set_ylabel("Count")
    ax.set_title("TopAneu-26: Slice Count Distribution")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(OUT_FIG/"slice_count_distribution.png"), dpi=120, bbox_inches="tight")
    plt.close(); print("Saved slice_count_distribution.png")

    # Intensity distribution (sampled)
    int_means = [float(r["int_mean"]) for r in img_rows]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(int_means, bins=40, color="#10B981", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean intensity (per volume)"); ax.set_ylabel("Count")
    ax.set_title("TopAneu-26: Per-Volume Mean Intensity Distribution")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(OUT_FIG/"intensity_distribution.png"), dpi=120, bbox_inches="tight")
    plt.close(); print("Saved intensity_distribution.png")

    # Anisotropy
    aniso = [float(r["anisotropy_ratio"]) for r in img_rows]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(aniso, bins=30, color="#F59E0B", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Anisotropy ratio (max_spacing / min_spacing)"); ax.set_ylabel("Count")
    ax.set_title("TopAneu-26: Voxel Anisotropy Distribution")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(OUT_FIG/"anisotropy_distribution.png"), dpi=120, bbox_inches="tight")
    plt.close(); print("Saved anisotropy_distribution.png")
else:
    print("imaging_characterization.csv not found — skipping distribution figures")

# Aneurysm location bar chart
aneu_csv = pathlib.Path(r"d:\NLP_Project\reports\data_survey\aneurysm_statistics.csv")
if aneu_csv.exists():
    with open(aneu_csv, encoding="utf-8") as f:
        rdr = csvmod.DictReader(f)
        aneu_rows = list(rdr)
    loc_counts = {}
    for r in aneu_rows:
        name = r["location_name"]
        loc_counts[name] = loc_counts.get(name, 0) + 1
    sorted_locs = sorted(loc_counts.items(), key=lambda x: -x[1])
    names = [x[0] for x in sorted_locs[:20]]
    counts = [x[1] for x in sorted_locs[:20]]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), counts, color="#EF4444")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Number of aneurysms")
    ax.set_title("TopAneu-26: Aneurysm Location Distribution (top 20)")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(OUT_FIG/"aneurysm_location_distribution.png"), dpi=120, bbox_inches="tight")
    plt.close(); print("Saved aneurysm_location_distribution.png")

print("\nAll figures done.")
