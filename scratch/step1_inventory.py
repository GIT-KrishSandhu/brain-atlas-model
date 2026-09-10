"""
STEP 1 — Filesystem inventory of topaneu_release/
Produces: data_inventory.csv, data_inventory.json, raw_inventory.txt
READ-ONLY. Does NOT modify any original files.
"""

import os, json, csv, hashlib, collections, pathlib, sys

DATASET_ROOT = pathlib.Path(r"d:\NLP_Project\topaneu_release")
OUT_DIR      = pathlib.Path(r"d:\NLP_Project\reports\data_survey")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Recursive walk ──────────────────────────────────────────────────────────
rows = []
ext_counter   = collections.Counter()
folder_counts = collections.defaultdict(int)
total_size    = 0

for root, dirs, files in os.walk(DATASET_ROOT):
    dirs.sort()
    for fn in sorted(files):
        fp  = pathlib.Path(root) / fn
        rel = fp.relative_to(DATASET_ROOT)
        try:
            size = fp.stat().st_size
        except Exception:
            size = -1
        ext  = "".join(fp.suffixes).lower()   # e.g. .nii.gz
        ext1 = fp.suffix.lower()               # last suffix only
        rows.append({
            "rel_path":    str(rel),
            "folder":      str(rel.parent),
            "filename":    fn,
            "ext":         ext,
            "size_bytes":  size,
            "size_kb":     round(size / 1024, 2),
            "size_mb":     round(size / (1024**2), 4),
        })
        ext_counter[ext] += 1
        folder_counts[str(rel.parent)] += 1
        total_size += size

# ── 2. Summary stats ───────────────────────────────────────────────────────────
summary = {
    "dataset_root":      str(DATASET_ROOT),
    "total_files":       len(rows),
    "total_size_bytes":  total_size,
    "total_size_mb":     round(total_size / (1024**2), 2),
    "total_size_gb":     round(total_size / (1024**3), 4),
    "extensions":        dict(ext_counter.most_common()),
    "files_per_folder":  dict(sorted(folder_counts.items())),
}

# ── 3. Per-folder breakdown by category ───────────────────────────────────────
categories = {
    "images":           [r for r in rows if r["folder"] == "images"],
    "vessel_masks":     [r for r in rows if r["folder"] == "vessel_masks"],
    "location_masks":   [r for r in rows if r["folder"] == "location_masks"],
    "type_masks":       [r for r in rows if r["folder"] == "type_masks"],
    "location_jsons":   [r for r in rows if r["folder"] == "location_jsons"],
    "root_files":       [r for r in rows if r["folder"] == "."],
}

for cat, items in categories.items():
    summary[f"count_{cat}"] = len(items)
    if items:
        sizes = [i["size_bytes"] for i in items if i["size_bytes"] >= 0]
        if sizes:
            summary[f"size_mb_{cat}_total"] = round(sum(sizes)/(1024**2), 2)
            summary[f"size_mb_{cat}_min"]   = round(min(sizes)/(1024**2), 4)
            summary[f"size_mb_{cat}_max"]   = round(max(sizes)/(1024**2), 4)
            summary[f"size_mb_{cat}_mean"]  = round((sum(sizes)/len(sizes))/(1024**2), 4)

# ── 4. Write CSV ───────────────────────────────────────────────────────────────
csv_path = OUT_DIR / "data_inventory.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# ── 5. Write JSON ──────────────────────────────────────────────────────────────
json_path = OUT_DIR / "data_inventory.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "files": rows}, f, indent=2)

# ── 6. Print raw report ────────────────────────────────────────────────────────
print("=" * 70)
print("DATASET INVENTORY — topaneu_release/")
print("=" * 70)
print(f"Root:             {DATASET_ROOT}")
print(f"Total files:      {len(rows)}")
print(f"Total size:       {summary['total_size_mb']} MB  ({summary['total_size_gb']} GB)")
print()
print("── Extensions ───")
for ext, cnt in ext_counter.most_common():
    print(f"  {ext or '(no ext)':20s}  {cnt:5d}")
print()
print("── Files per folder ──")
for fld, cnt in sorted(folder_counts.items()):
    print(f"  {fld or '.':30s}  {cnt:5d}")
print()
print("── Category counts ──")
for cat in ["images","vessel_masks","location_masks","type_masks","location_jsons","root_files"]:
    cnt  = summary.get(f"count_{cat}", 0)
    stot = summary.get(f"size_mb_{cat}_total", 0)
    print(f"  {cat:20s}  {cnt:5d} files   {stot:8.2f} MB")
print()

# ── 7. Identify naming pattern ─────────────────────────────────────────────────
img_names  = [r["filename"] for r in categories["images"]]
msk_names  = [r["filename"] for r in categories["vessel_masks"]]
loc_names  = [r["filename"] for r in categories["location_masks"]]
typ_names  = [r["filename"] for r in categories["type_masks"]]
jsn_names  = [r["filename"] for r in categories["location_jsons"]]

print("── Sample filenames (first 8 of each) ──")
for label, lst in [("images", img_names), ("vessel_masks", msk_names),
                   ("location_masks", loc_names), ("type_masks", typ_names),
                   ("location_jsons", jsn_names)]:
    print(f"  {label}:")
    for n in lst[:8]:
        print(f"    {n}")
print()

print(f"Outputs written to: {OUT_DIR}")
