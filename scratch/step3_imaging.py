"""
STEP 3 — NIfTI imaging characterization
READ-ONLY. Writes stats only to reports/data_survey/
Checks nibabel availability first.
"""
import sys, pathlib, json, csv, collections

try:
    import nibabel as nib
    import numpy as np
    print("nibabel OK:", nib.__version__)
    print("numpy OK:", np.__version__)
except ImportError as e:
    print("MISSING:", e)
    sys.exit(1)

ROOT   = pathlib.Path(r"d:\NLP_Project\topaneu_release")
OUT    = pathlib.Path(r"d:\NLP_Project\reports\data_survey")
IMG_DIR = ROOT / "images"

real_imgs = sorted([f for f in IMG_DIR.iterdir() if not f.name.startswith("._")])
print(f"\nReal imaging volumes to scan: {len(real_imgs)}")

records = []
errors  = []

for i, fp in enumerate(real_imgs):
    if i % 50 == 0:
        print(f"  Processing {i}/{len(real_imgs)} ...")
    try:
        img   = nib.load(str(fp))
        hdr   = img.header
        shape = img.shape
        zoom  = img.header.get_zooms()
        dtype = str(img.get_data_dtype())

        # Load data for intensity stats
        data  = img.get_fdata(dtype=np.float32)
        dmin  = float(data.min())
        dmax  = float(data.max())
        dmean = float(data.mean())
        dstd  = float(data.std())
        p1    = float(np.percentile(data, 1))
        p5    = float(np.percentile(data, 5))
        p95   = float(np.percentile(data, 95))
        p99   = float(np.percentile(data, 99))
        has_nan = bool(np.any(np.isnan(data)))
        has_inf = bool(np.any(np.isinf(data)))

        ndim  = len(shape)
        sx, sy, sz = (shape[0], shape[1], shape[2]) if ndim >= 3 else (0,0,0)
        vx = float(zoom[0]) if len(zoom) > 0 else 0
        vy = float(zoom[1]) if len(zoom) > 1 else 0
        vz = float(zoom[2]) if len(zoom) > 2 else 0
        fov_x = round(sx * vx, 2)
        fov_y = round(sy * vy, 2)
        fov_z = round(sz * vz, 2)
        anisotropy = round(max(vx,vy,vz) / (min(vx,vy,vz)+1e-9), 4)

        records.append({
            "case_id":   fp.stem.replace("_0000",""),
            "filename":  fp.name,
            "shape_x": sx, "shape_y": sy, "shape_z": sz,
            "ndim": ndim,
            "voxel_x_mm": round(vx,4), "voxel_y_mm": round(vy,4), "voxel_z_mm": round(vz,4),
            "fov_x_mm": fov_x, "fov_y_mm": fov_y, "fov_z_mm": fov_z,
            "anisotropy_ratio": anisotropy,
            "dtype": dtype,
            "int_min": round(dmin,4), "int_max": round(dmax,4),
            "int_mean": round(dmean,4), "int_std": round(dstd,4),
            "p1": round(p1,4), "p5": round(p5,4), "p95": round(p95,4), "p99": round(p99,4),
            "has_nan": has_nan, "has_inf": has_inf,
        })
    except Exception as e:
        errors.append({"filename": fp.name, "error": str(e)})
        print(f"  ERROR: {fp.name}: {e}")

print(f"\nSuccessfully read: {len(records)}")
print(f"Errors: {len(errors)}")

# Write per-case CSV
csv_path = OUT / "imaging_characterization.csv"
if records:
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader(); w.writerows(records)
    print(f"Wrote {csv_path}")

# Summary stats
if records:
    arr = lambda key: [r[key] for r in records]
    def stats(vals):
        a = np.array(vals, dtype=float)
        return {"min":round(float(a.min()),4),"max":round(float(a.max()),4),
                "mean":round(float(a.mean()),4),"std":round(float(a.std()),4),
                "unique_count": int(len(set(vals)))}

    summary = {
        "n_volumes": len(records),
        "shape_x": stats(arr("shape_x")),
        "shape_y": stats(arr("shape_y")),
        "shape_z": stats(arr("shape_z")),
        "voxel_x_mm": stats(arr("voxel_x_mm")),
        "voxel_y_mm": stats(arr("voxel_y_mm")),
        "voxel_z_mm": stats(arr("voxel_z_mm")),
        "anisotropy_ratio": stats(arr("anisotropy_ratio")),
        "int_min": stats(arr("int_min")),
        "int_max": stats(arr("int_max")),
        "int_mean": stats(arr("int_mean")),
        "int_std":  stats(arr("int_std")),
        "dtypes": dict(collections.Counter(arr("dtype"))),
        "has_nan_count": sum(arr("has_nan")),
        "has_inf_count": sum(arr("has_inf")),
        "errors": errors,
        "unique_shapes": list(set([(r["shape_x"],r["shape_y"],r["shape_z"]) for r in records]))[:20],
        "unique_voxel_spacings": list(set([(r["voxel_x_mm"],r["voxel_y_mm"],r["voxel_z_mm"]) for r in records]))[:20],
    }
    with open(OUT/"imaging_summary.json","w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nIMAGING SUMMARY:")
    print(json.dumps(summary, indent=2))
