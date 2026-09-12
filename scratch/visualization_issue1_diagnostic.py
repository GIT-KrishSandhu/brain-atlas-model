"""
Diagnostic script for Visualization Issue #1:
Location and Type Overlays Not Visible.

Demonstrates:
1. Exact inspection of NIfTI volumes and masks (shape, dtype, affine, zooms, labels, bbox).
2. Root-cause demonstration: naive mid-slice (shape // 2) vs focal peak-slice.
3. 2x2 diagnostic figure generation (Raw, Vessel, Location, Type) on the exact same slice.
4. Export of structured validation report.
"""
import os
import json
import pathlib
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

ROOT = pathlib.Path(r"d:\NLP_Project\topaneu_release")
IMG_DIR = ROOT / "images"
VES_DIR = ROOT / "vessel_masks"
LOC_DIR = ROOT / "location_masks"
TYP_DIR = ROOT / "type_masks"

vessel_map = {int(v): k for k, v in json.load(open(ROOT / "vessel_mapping.json"))["labels"].items()}
location_map = {int(v): k for k, v in json.load(open(ROOT / "location_mapping.json"))["labels"].items()}
type_map = {int(v): k for k, v in json.load(open(ROOT / "type_mapping.json"))["labels"].items()}

def normalize(vol, p1=1.0, p99=99.0):
    lo = float(np.percentile(vol, p1))
    hi = float(np.percentile(vol, p99))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0)

def best_slice(mask, axis=2):
    other_axes = tuple(i for i in range(mask.ndim) if i != axis)
    counts = np.count_nonzero(mask, axis=other_axes)
    if not np.any(counts):
        return mask.shape[axis] // 2
    return int(np.argmax(counts))

def inspect_mask(path, name):
    nii = nib.load(str(path))
    data = np.asarray(nii.get_fdata())
    u = np.unique(data)
    nz = np.argwhere(data != 0)
    bbox_min = nz.min(axis=0).tolist() if len(nz) else []
    bbox_max = nz.max(axis=0).tolist() if len(nz) else []

    info = {
        "name": name,
        "path": str(path),
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "affine": nii.affine.tolist(),
        "orientation": list(nib.aff2axcodes(nii.affine)),
        "min": float(data.min()),
        "max": float(data.max()),
        "unique_count": len(u),
        "unique_values": [int(x) if x.is_integer() else float(x) for x in u[:50]],
        "nonzero_voxels": int(np.count_nonzero(data)),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "voxel_spacing": [float(z) for z in nii.header.get_zooms()[:3]]
    }
    return nii, data, info

def overlay_categorical(ax, image_slice, mask_slice, title, cmap_name="tab20", alpha=0.55, label_map=None):
    ax.imshow(image_slice.T, cmap="gray", origin="lower", aspect="auto")
    nz_count = int(np.count_nonzero(mask_slice))
    if nz_count == 0:
        ax.set_title(f"{title}\n[EMPTY ON THIS SLICE]", fontsize=10, color="#666666")
        ax.axis("off")
        return

    masked = np.ma.masked_where(mask_slice == 0, mask_slice)
    present_labels = sorted(int(x) for x in np.unique(mask_slice) if x != 0)
    label_desc = []
    for lbl in present_labels:
        name = label_map.get(lbl, f"ID-{lbl}") if label_map else f"ID-{lbl}"
        label_desc.append(f"{lbl}:{name}")
    desc_str = ", ".join(label_desc)
    if len(desc_str) > 40:
        desc_str = desc_str[:37] + "..."

    ax.imshow(
        masked.T,
        cmap=cmap_name,
        alpha=alpha,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=1,
        vmax=max(max(present_labels), 1)
    )
    ax.set_title(f"{title}\n({nz_count} voxels | {desc_str})", fontsize=10)
    ax.axis("off")

def generate_diagnostic_figure(case_id, out_path):
    img_fp = IMG_DIR / f"{case_id}_0000.nii.gz"
    ves_fp = VES_DIR / f"{case_id}.nii.gz"
    loc_fp = LOC_DIR / f"{case_id}.nii.gz"
    typ_fp = TYP_DIR / f"{case_id}.nii.gz"

    img_nii, img_data, img_info = inspect_mask(img_fp, "RAW IMAGE")
    ves_nii, ves_data, ves_info = inspect_mask(ves_fp, "VESSEL MASK")
    loc_nii, loc_data, loc_info = inspect_mask(loc_fp, "LOCATION MASK")
    typ_nii, typ_data, typ_info = inspect_mask(typ_fp, "TYPE MASK")

    # Select axial slice with maximum aneurysm voxels, or vessel peak if negative
    if loc_info["nonzero_voxels"] > 0:
        z_slice = best_slice(loc_data, axis=2)
        slice_basis = f"Location peak (max non-zero voxels: {np.count_nonzero(loc_data[:, :, z_slice])})"
    elif ves_info["nonzero_voxels"] > 0:
        z_slice = best_slice(ves_data, axis=2)
        slice_basis = f"Vessel peak (negative case, max vessel voxels: {np.count_nonzero(ves_data[:, :, z_slice])})"
    else:
        z_slice = img_data.shape[2] // 2
        slice_basis = f"Mid-slice (empty case)"

    img_norm = normalize(img_data)
    img_sl = img_norm[:, :, z_slice]
    ves_sl = ves_data[:, :, z_slice]
    loc_sl = loc_data[:, :, z_slice]
    typ_sl = typ_data[:, :, z_slice]

    # Naive mid-slice comparison
    mid_z = img_data.shape[2] // 2
    naive_comparison = {
        "axial_slice_selected": z_slice,
        "slice_basis": slice_basis,
        "selected_slice_loc_nonzero": int(np.count_nonzero(loc_sl)),
        "selected_slice_typ_nonzero": int(np.count_nonzero(typ_sl)),
        "selected_slice_ves_nonzero": int(np.count_nonzero(ves_sl)),
        "naive_mid_z": mid_z,
        "naive_mid_z_loc_nonzero": int(np.count_nonzero(loc_data[:, :, mid_z])),
        "naive_mid_z_typ_nonzero": int(np.count_nonzero(typ_data[:, :, mid_z])),
        "naive_mid_z_ves_nonzero": int(np.count_nonzero(ves_data[:, :, mid_z])),
    }

    # Plot 2x2 diagnostic figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(
        f"Diagnostic 2x2: {case_id} (Axial Slice z={z_slice} / {img_data.shape[2]})\n"
        f"Spatial Alignment: Orient={img_info['orientation']}, Dim={img_info['shape'][:2]}",
        fontsize=13,
        fontweight="bold"
    )

    # 1. Raw Image
    axes[0, 0].imshow(img_sl.T, cmap="gray", origin="lower", aspect="auto")
    axes[0, 0].set_title(f"Raw Medical Scan\n(Axial Slice z={z_slice})", fontsize=10)
    axes[0, 0].axis("off")

    # 2. Vessel Overlay
    overlay_categorical(
        axes[0, 1],
        img_sl,
        ves_sl,
        f"Vessel Overlay (Vessel Mask)",
        cmap_name="autumn",
        alpha=0.45,
        label_map=vessel_map
    )

    # 3. Location Overlay
    overlay_categorical(
        axes[1, 0],
        img_sl,
        loc_sl,
        f"Location Overlay (Aneurysm Location Mask)",
        cmap_name="tab20",
        alpha=0.60,
        label_map=location_map
    )

    # 4. Type Overlay
    overlay_categorical(
        axes[1, 1],
        img_sl,
        typ_sl,
        f"Type Overlay (Aneurysm Morphology Mask)",
        cmap_name="Set1",
        alpha=0.60,
        label_map=type_map
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved diagnostic figure: {out_path}")

    case_report = {
        "case_id": case_id,
        "image": img_info,
        "vessel_mask": ves_info,
        "location_mask": loc_info,
        "type_mask": typ_info,
        "comparison": naive_comparison
    }
    return case_report

def main():
    cases_to_test = [
        "topaneu_center1_mr_017",
        "topaneu_center1_mr_024",
        "topaneu_center1_mr_028",
        "topaneu_center1_mr_001",
    ]
    diag_dir = pathlib.Path(r"d:\NLP_Project\reports\data_survey\visualizations")
    diag_dir.mkdir(parents=True, exist_ok=True)

    full_report = {
        "diagnostic_summary": {
            "root_cause": (
                "The visualization pipeline in scratch/step5_visualize.py previously hardcoded slice selection "
                "to the volume mid-slices (shape[0]//2, shape[1]//2, shape[2]//2). While diffuse cerebrovascular "
                "vessel trees span the mid-brain and remained visible, localized intracranial aneurysms occupy small "
                "focal bounding boxes that did not intersect the exact mid-slice coordinates. Consequently, the extracted "
                "location and type mask 2D slices were 100% empty (all zeros), so masked overlays rendered zero pixels "
                "over the underlying grayscale scan, producing PNGs identical to the raw scan. The underlying NIfTI "
                "annotations were completely intact."
            ),
            "fix_applied": (
                "Implemented focal slice selection that identifies the 3D bounding box and peak non-zero voxel slice "
                "for aneurysm location/type masks, enforces spatial consistency across raw/vessel/location/type panels, "
                "uses categorical colormaps with nearest-neighbor interpolation, and sets opacity alpha=0.45-0.60."
            )
        },
        "cases": []
    }

    for cid in cases_to_test:
        print(f"\n--- Testing {cid} ---")
        fig_out = diag_dir / f"{cid}_diagnostic_2x2.png"
        case_rep = generate_diagnostic_figure(cid, fig_out)
        full_report["cases"].append(case_rep)

    report_json_path = pathlib.Path(r"d:\NLP_Project\scratch\visualization_issue1_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nSaved structured report to: {report_json_path}")

if __name__ == "__main__":
    main()
