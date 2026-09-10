"""
STEP 4 — Mask analysis: vessel, location, type masks
READ-ONLY. Writes to reports/data_survey/ only.
"""
import pathlib, json, csv, collections, sys
import nibabel as nib
import numpy as np

ROOT      = pathlib.Path(r"d:\NLP_Project\topaneu_release")
OUT       = pathlib.Path(r"d:\NLP_Project\reports\data_survey")
IMG_DIR   = ROOT / "images"
VES_DIR   = ROOT / "vessel_masks"
LOC_DIR   = ROOT / "location_masks"
TYP_DIR   = ROOT / "type_masks"

# Load mapping files
vessel_map   = {v:k for k,v in json.load(open(ROOT/"vessel_mapping.json"))["labels"].items()}
location_map = {v:k for k,v in json.load(open(ROOT/"location_mapping.json"))["labels"].items()}
type_map     = {v:k for k,v in json.load(open(ROOT/"type_mapping.json"))["labels"].items()}

real_imgs = sorted([f for f in IMG_DIR.iterdir() if not f.name.startswith("._")])
print(f"Cases to process: {len(real_imgs)}")

records = []
annotation_summary_rows = []
quality_issues = []

for i, img_fp in enumerate(real_imgs):
    if i % 50 == 0:
        print(f"  {i}/{len(real_imgs)} ...")

    case_id = img_fp.stem.replace("_0000","")

    # Corresponding mask paths
    ves_fp = VES_DIR / f"{case_id}.nii.gz"
    loc_fp = LOC_DIR / f"{case_id}.nii.gz"
    typ_fp = TYP_DIR / f"{case_id}.nii.gz"

    try:
        img      = nib.load(str(img_fp))
        img_shp  = img.shape
        img_zoom = img.header.get_zooms()[:3]
    except Exception as e:
        quality_issues.append({"case_id":case_id,"file":img_fp.name,"problem":f"Cannot load image: {e}","severity":"CRITICAL"})
        continue

    row = {"case_id": case_id, "img_shape": str(img_shp[:3]), "img_zoom": str(tuple(round(z,4) for z in img_zoom))}

    for mask_name, mask_fp, mmap in [
        ("vessel_mask", ves_fp, vessel_map),
        ("location_mask", loc_fp, location_map),
        ("type_mask", typ_fp, type_map),
    ]:
        if not mask_fp.exists():
            row[f"{mask_name}_exists"] = False
            row[f"{mask_name}_shape_match"] = "MISSING"
            quality_issues.append({"case_id":case_id,"file":mask_fp.name,"problem":"Mask file missing","severity":"CRITICAL"})
            continue
        try:
            msk      = nib.load(str(mask_fp))
            msk_shp  = msk.shape
            msk_zoom = msk.header.get_zooms()[:3]
            msk_data = msk.get_fdata(dtype=np.float32)
            unique_labels = sorted(set(msk_data.flatten().astype(int).tolist()))
            n_nonzero = int(np.count_nonzero(msk_data))
            is_empty  = n_nonzero == 0

            shape_match = (msk_shp[:3] == img_shp[:3])
            zoom_match  = all(abs(a-b)<0.01 for a,b in zip(msk_zoom, img_zoom))

            row[f"{mask_name}_exists"]       = True
            row[f"{mask_name}_shape"]        = str(msk_shp[:3])
            row[f"{mask_name}_shape_match"]  = shape_match
            row[f"{mask_name}_zoom_match"]   = zoom_match
            row[f"{mask_name}_unique_labels"]= str(unique_labels)
            row[f"{mask_name}_n_labels"]     = len(unique_labels)
            row[f"{mask_name}_nonzero_vox"]  = n_nonzero
            row[f"{mask_name}_is_empty"]     = is_empty

            unexpected = [l for l in unique_labels if l not in mmap and l != 0]
            if unexpected:
                quality_issues.append({"case_id":case_id,"file":mask_fp.name,
                    "problem":f"Unexpected label IDs: {unexpected}","severity":"WARNING"})
            if not shape_match:
                quality_issues.append({"case_id":case_id,"file":mask_fp.name,
                    "problem":f"Shape mismatch: img={img_shp[:3]} mask={msk_shp[:3]}","severity":"CRITICAL"})
            if is_empty and mask_name in ["vessel_mask","location_mask"]:
                quality_issues.append({"case_id":case_id,"file":mask_fp.name,
                    "problem":"Empty mask (all zeros)","severity":"WARNING"})
        except Exception as e:
            row[f"{mask_name}_exists"] = True
            row[f"{mask_name}_shape_match"] = f"ERROR: {e}"
            quality_issues.append({"case_id":case_id,"file":mask_fp.name,"problem":str(e),"severity":"CRITICAL"})

    records.append(row)

print(f"\nProcessed: {len(records)}")
print(f"Quality issues found: {len(quality_issues)}")

# Write case_relationships.csv
with open(OUT/"case_relationships.csv","w",newline="",encoding="utf-8") as f:
    if records:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader(); w.writerows(records)
print(f"Wrote case_relationships.csv")

# Write data_quality_report.csv
issue_fields = ["case_id","file","problem","severity"]
with open(OUT/"data_quality_report.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=issue_fields)
    w.writeheader()
    for issue in quality_issues:
        w.writerow({k: issue.get(k,"") for k in issue_fields})
print(f"Wrote data_quality_report.csv  ({len(quality_issues)} issues)")

# Summary
shape_mismatches = [i for i in quality_issues if "Shape mismatch" in i["problem"]]
empty_masks      = [i for i in quality_issues if "Empty mask" in i["problem"]]
missing_masks    = [i for i in quality_issues if "missing" in i["problem"]]
print(f"\nShape mismatches: {len(shape_mismatches)}")
print(f"Empty masks:      {len(empty_masks)}")
print(f"Missing masks:    {len(missing_masks)}")
print(f"Other issues:     {len(quality_issues)-len(shape_mismatches)-len(empty_masks)-len(missing_masks)}")

# Type mask label distribution across dataset
print("\nTYPE_MASK_LABEL_DISTRIBUTION (aneurysm type annotations):")
type_counter = collections.Counter()
for r in records:
    ul = r.get("type_mask_unique_labels","[]")
    try:
        labels = [int(x) for x in ul.strip("[]").split(",") if x.strip()]
        for l in labels:
            if l != 0:
                type_counter[l] += 1
    except:
        pass
for tid, cnt in sorted(type_counter.items()):
    print(f"  Type {tid} ({type_map.get(tid,'?')}): {cnt} cases")
