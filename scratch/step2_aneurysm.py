import json, pathlib, collections, csv

ROOT = pathlib.Path(r"d:\NLP_Project\topaneu_release")
OUT  = pathlib.Path(r"d:\NLP_Project\reports\data_survey")
jsn_dir = ROOT / "location_jsons"
real_jsons = sorted([f for f in jsn_dir.iterdir() if not f.name.startswith("._")])

loc_map_raw = json.load(open(ROOT/"location_mapping.json"))["labels"]
loc_map = {v: k for k, v in loc_map_raw.items()}
type_map_raw = json.load(open(ROOT/"type_mapping.json"))["labels"]

positive_cases = []
negative_cases = []
all_locations  = []
multi_aneurysm = []

for jp in real_jsons:
    with open(jp) as f:
        data = json.load(f)
    locs = data.get("locations", [])
    if locs:
        positive_cases.append({"case": jp.stem, "locations": locs, "n_aneurysms": len(locs)})
        all_locations.extend(locs)
        if len(locs) > 1:
            multi_aneurysm.append({"case": jp.stem, "n": len(locs), "locations": locs})
    else:
        negative_cases.append(jp.stem)

print("TOTAL_CASES:", len(real_jsons))
print("POSITIVE_CASES:", len(positive_cases))
print("NEGATIVE_CASES:", len(negative_cases))
print("MULTI_ANEURYSM_CASES:", len(multi_aneurysm))
print("TOTAL_ANEURYSMS:", len(all_locations))

loc_counter = collections.Counter(all_locations)
print("\nANEURYSM_LOCATION_DISTRIBUTION:")
for loc_id, cnt in sorted(loc_counter.items(), key=lambda x: -x[1]):
    name = loc_map.get(loc_id, f"UNKNOWN_{loc_id}")
    print(f"  Label {loc_id:3d}  {name}: {cnt}")

n_per_case = [p["n_aneurysms"] for p in positive_cases]
n_dist = collections.Counter(n_per_case)
print("\nANEURYSMS_PER_POSITIVE_CASE:", dict(sorted(n_dist.items())))

print("\nMULTI_ANEURYSM_CASES:")
for c in multi_aneurysm[:15]:
    lnames = [loc_map.get(l, str(l)) for l in c["locations"]]
    print(f"  {c['case']}: n={c['n']}  {lnames}")

# Write aneurysm_statistics.csv
rows = []
for c in positive_cases:
    for loc_id in c["locations"]:
        rows.append({"case_id": c["case"], "location_label_id": loc_id,
                     "location_name": loc_map.get(loc_id,"UNKNOWN"),
                     "n_aneurysms_in_case": c["n_aneurysms"]})

with open(OUT/"aneurysm_statistics.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["case_id","location_label_id","location_name","n_aneurysms_in_case"])
    w.writeheader(); w.writerows(rows)
print(f"\nWrote aneurysm_statistics.csv  ({len(rows)} rows)")

# Class imbalance
total = len(real_jsons)
pos   = len(positive_cases)
neg   = len(negative_cases)
print(f"\nCLASS_IMBALANCE: pos={pos} ({100*pos/total:.1f}%)  neg={neg} ({100*neg/total:.1f}%)  ratio={neg/pos:.2f}:1")
