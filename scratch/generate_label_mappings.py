import os
import json
import csv

TOPANEU_DIR = r"D:\NLP_Project\topaneu_release"
CSV_PATH = r"D:\NLP_Project\PHASE2_LABEL_MAPPING.csv"
MD_PATH = r"D:\NLP_Project\PHASE2_LABEL_MAPPING.md"

with open(os.path.join(TOPANEU_DIR, "location_mapping.json")) as f:
    loc_labels = json.load(f)["labels"]

with open(os.path.join(TOPANEU_DIR, "vessel_mapping.json")) as f:
    vess_labels = json.load(f)["labels"]

# P3 Locations (0 to 12 in cls head, 1 to 13 in dataset.json)
# dataset.json IDs:
# 1: Other Posterior Circulation
# 2: BA-Tip
# 3: Right Posterior Communicating Artery
# 4: Left Posterior Communicating Artery
# 5: Right Infraclinoid Internal Carotid Artery
# 6: Left Infraclinoid Internal Carotid Artery
# 7: Right Supraclinoid Internal Carotid Artery
# 8: Left Supraclinoid Internal Carotid Artery
# 9: Right Middle Cerebral Artery
# 10: Left Middle Cerebral Artery
# 11: Right Anterior Cerebral Artery
# 12: Left Anterior Cerebral Artery
# 13: Anterior Communicating Artery

# Comprehensive mapping for all 52 TopAneu Location Labels
# Mapping types: EXACT, MERGED, SPLIT, RELATED, NO_DIRECT_EQUIVALENT
mapping_rules = []

for name, tid in loc_labels.items():
    if tid == 0:
        continue
    
    # Defaults
    p3_id = None
    p3_name = None
    m_type = "RELATED"
    conf = "HIGH"
    notes = ""

    # Vertebrobasilar & Cerebellar
    if "1.1 VA trunk" in name or "1.2 PICA trunk" in name or "1.3 VA-PICA junction" in name or "1.4 BA trunk" in name or "1.5 VA-BA junction" in name or "1.6 AICA trunk" in name or "1.7 BA-AICA junction" in name or "1.8 SCA trunk" in name or "1.9 BA-SCA junction" in name:
        p3_id = 1
        p3_name = "Other Posterior Circulation"
        m_type = "MERGED"
        conf = "HIGH"
        notes = "P3 merges all non-apex posterior circulation arteries (VA, BA trunk, PICA, AICA, SCA) into Other Posterior Circulation."
    elif "1.10 BA tip" in name:
        p3_id = 2
        p3_name = "Basilar Tip"
        m_type = "EXACT"
        conf = "HIGH"
        notes = "Exact anatomical match for basilar bifurcation / apex aneurysm."
    elif "2.1 P1P2" in name or "2.2 P3P4" in name:
        p3_id = 1
        p3_name = "Other Posterior Circulation"
        m_type = "MERGED"
        conf = "HIGH"
        notes = "TopAneu distinguishes PCA segments (P1-P4); P3 groups all PCA branches into Other Posterior Circulation."
    elif "3.1 ICA infraclinoid C1-C5" in name:
        if name.startswith("R-"):
            p3_id = 5
            p3_name = "Right Infraclinoid Internal Carotid Artery"
        else:
            p3_id = 6
            p3_name = "Left Infraclinoid Internal Carotid Artery"
        m_type = "EXACT"
        conf = "HIGH"
        notes = "Exact anatomical match for cervical/petrous/cavernous/clinoid segments."
    elif "3.2 ICA C6-OA-junction" in name or "3.3 ICA C6-nonOA" in name or "3.6 ICA C7-nonBranch" in name or "3.7 ICA C7-terminus" in name:
        if name.startswith("R-"):
            p3_id = 7
            p3_name = "Right Supraclinoid Internal Carotid Artery"
        else:
            p3_id = 8
            p3_name = "Left Supraclinoid Internal Carotid Artery"
        m_type = "MERGED"
        conf = "HIGH"
        notes = "TopAneu subdivides supraclinoid ICA into C6-OA, C6-nonOA, C7-nonBranch, and C7-terminus; P3 combines these into Supraclinoid ICA."
    elif "3.4 ICA C7-Pcom-junction" in name:
        # Note: TopAneu labels this as ICA C7 Pcom junction. In P3, Pcom has a dedicated class.
        if name.startswith("R-"):
            p3_id = 3
            p3_name = "Right Posterior Communicating Artery"
        else:
            p3_id = 4
            p3_name = "Left Posterior Communicating Artery"
        m_type = "RELATED"
        conf = "MEDIUM"
        notes = "TopAneu considers this the ICA junction; P3 annotates PCom aneurysms under Posterior Communicating Artery. Also closely related to Supraclinoid ICA."
    elif "3.5 ICA C7-AChA-junction" in name:
        if name.startswith("R-"):
            p3_id = 7
            p3_name = "Right Supraclinoid Internal Carotid Artery"
        else:
            p3_id = 8
            p3_name = "Left Supraclinoid Internal Carotid Artery"
        m_type = "MERGED"
        conf = "HIGH"
        notes = "Anterior choroidal junction is part of C7 supraclinoid segment in P3 taxonomy."
    elif "4.1 Acom complex" in name:
        p3_id = 13
        p3_name = "Anterior Communicating Artery"
        m_type = "EXACT"
        conf = "HIGH"
        notes = "Exact anatomical match for ACom complex aneurysms."
    elif "4.2 A1" in name or "4.3 A2" in name or "4.4 A3" in name or "4.5 Distal ACA branches" in name:
        if name.startswith("R-"):
            p3_id = 11
            p3_name = "Right Anterior Cerebral Artery"
        else:
            p3_id = 12
            p3_name = "Left Anterior Cerebral Artery"
        m_type = "MERGED"
        conf = "HIGH"
        notes = "TopAneu divides ACA into A1, A2, A3, and distal branches; P3 pools all ipsilateral ACA segments into Right/Left ACA."
    elif "5.1 M1 trunk" in name or "5.2 M1 early bifurcation" in name or "5.3 M1-M2 junction" in name or "Distal-M2M3" in name:
        if name.startswith("R-"):
            p3_id = 9
            p3_name = "Right Middle Cerebral Artery"
        else:
            p3_id = 10
            p3_name = "Left Middle Cerebral Artery"
        m_type = "MERGED"
        conf = "HIGH"
        notes = "TopAneu divides MCA into M1 trunk, bifurcation, junction, and distal branches; P3 pools all into Right/Left MCA."
    else:
        m_type = "NO_DIRECT_EQUIVALENT"
        conf = "LOW"
        notes = "No direct corresponding class in P3 13-class taxonomy."

    mapping_rules.append({
        "category": "Location (Aneurysm)",
        "topaneu_id": tid,
        "topaneu_name": name,
        "p3_id": p3_id if p3_id else "N/A",
        "p3_name": p3_name if p3_name else "NO_DIRECT_EQUIVALENT",
        "mapping_type": m_type,
        "confidence": conf,
        "notes": notes
    })

# Vessel Mapping (36 labels)
for name, tid in vess_labels.items():
    if tid == 0:
        continue
    p3_id = None
    p3_name = None
    m_type = "RELATED"
    conf = "HIGH"
    notes = ""

    if name in ["BA", "R-VA", "L-VA", "R-P1P2", "L-P1P2", "R-P3P4", "L-P3P4", "R-SCA", "L-SCA", "R-AICA", "L-AICA", "R-PICA", "L-PICA"]:
        if name == "BA":
            p3_id = 1  # Note: BA trunk vs BA tip
            p3_name = "Other Posterior Circulation"
            m_type = "RELATED"
            conf = "HIGH"
            notes = "Basilar artery trunk maps to Other Posterior Circulation in P3; apex maps to Basilar Tip."
        else:
            p3_id = 1
            p3_name = "Other Posterior Circulation"
            m_type = "MERGED"
            conf = "HIGH"
            notes = "Vertebral, cerebellar, and posterior cerebral arteries merged into Other Posterior Circulation."
    elif name == "R-ICA-C1-C5":
        p3_id = 5; p3_name = "Right Infraclinoid Internal Carotid Artery"; m_type = "EXACT"; conf = "HIGH"; notes = "Exact match for infraclinoid ICA."
    elif name == "L-ICA-C1-C5":
        p3_id = 6; p3_name = "Left Infraclinoid Internal Carotid Artery"; m_type = "EXACT"; conf = "HIGH"; notes = "Exact match for infraclinoid ICA."
    elif name == "R-ICA-C6-C7":
        p3_id = 7; p3_name = "Right Supraclinoid Internal Carotid Artery"; m_type = "EXACT"; conf = "HIGH"; notes = "Exact match for supraclinoid ICA."
    elif name == "L-ICA-C6-C7":
        p3_id = 8; p3_name = "Left Supraclinoid Internal Carotid Artery"; m_type = "EXACT"; conf = "HIGH"; notes = "Exact match for supraclinoid ICA."
    elif name in ["R-M1", "R-M2", "R-M3"]:
        p3_id = 9; p3_name = "Right Middle Cerebral Artery"; m_type = "MERGED"; conf = "HIGH"; notes = "M1, M2, M3 combined into Right MCA."
    elif name in ["L-M1", "L-M2", "L-M3"]:
        p3_id = 10; p3_name = "Left Middle Cerebral Artery"; m_type = "MERGED"; conf = "HIGH"; notes = "M1, M2, M3 combined into Left MCA."
    elif name in ["R-A1A2", "R-A3"]:
        p3_id = 11; p3_name = "Right Anterior Cerebral Artery"; m_type = "MERGED"; conf = "HIGH"; notes = "A1, A2, A3 combined into Right ACA."
    elif name in ["L-A1A2", "L-A3"]:
        p3_id = 12; p3_name = "Left Anterior Cerebral Artery"; m_type = "MERGED"; conf = "HIGH"; notes = "A1, A2, A3 combined into Left ACA."
    elif name in ["3rd-A2", "3rd-A3"]:
        p3_id = "N/A"; p3_name = "NO_DIRECT_EQUIVALENT"; m_type = "NO_DIRECT_EQUIVALENT"; conf = "HIGH"; notes = "Anomalous median anterior cerebral artery variant; not present in P3 standard anatomy."
    elif name == "Acom":
        p3_id = 13; p3_name = "Anterior Communicating Artery"; m_type = "EXACT"; conf = "HIGH"; notes = "Exact match for ACom artery."
    elif name == "R-Pcom":
        p3_id = 3; p3_name = "Right Posterior Communicating Artery"; m_type = "EXACT"; conf = "HIGH"; notes = "Exact match for right PCom."
    elif name == "L-Pcom":
        p3_id = 4; p3_name = "Left Posterior Communicating Artery"; m_type = "EXACT"; conf = "HIGH"; notes = "Exact match for left PCom."
    elif name in ["R-AChA", "L-AChA", "R-OA", "L-OA"]:
        p3_id = 7 if name.startswith("R-") else 8
        p3_name = "Right Supraclinoid Internal Carotid Artery" if name.startswith("R-") else "Left Supraclinoid Internal Carotid Artery"
        m_type = "RELATED"
        conf = "MEDIUM"
        notes = "Small anterior choroidal and ophthalmic branches arise from C6/C7 ICA; P3 does not separate micro-branches."

    mapping_rules.append({
        "category": "Vessel (Anatomy)",
        "topaneu_id": tid,
        "topaneu_name": name,
        "p3_id": p3_id if p3_id else "N/A",
        "p3_name": p3_name if p3_name else "NO_DIRECT_EQUIVALENT",
        "mapping_type": m_type,
        "confidence": conf,
        "notes": notes
    })

# Write CSV
fieldnames = ["category", "topaneu_id", "topaneu_name", "p3_id", "p3_name", "mapping_type", "confidence", "notes"]
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(mapping_rules)
print(f"Saved {len(mapping_rules)} mapping rules to: {CSV_PATH}")

# Write Markdown
with open(MD_PATH, "w", encoding="utf-8") as f:
    f.write("# Phase 2: Explicit TopAneu ↔ P3 Semantic & Anatomical Mapping Matrix\n\n")
    f.write("**Project:** Brain Atlas / Intracranial Aneurysm Detection  \n")
    f.write("**Evaluation:** TopAneu (52 Locations, 36 Vessels) to P3 (13 Arterial Classes) Alignment  \n")
    f.write("**Date:** September 12, 2026  \n\n")
    f.write("---\n\n")
    f.write("## 1. Mapping Methodology & Type Definitions\n\n")
    f.write("To prevent artificial inflation of alignment metrics, mappings are strictly categorized into five mutually exclusive types:\n")
    f.write("- **EXACT:** Direct one-to-one anatomical equivalence (e.g., TopAneu `1.10 BA tip` $\\leftrightarrow$ P3 `Basilar Tip`).\n")
    f.write("- **MERGED:** Multiple fine-grained TopAneu structures correspond to a single broader P3 territory (e.g., TopAneu `R-M1`, `R-M2`, `R-M3` $\\to$ P3 `Right MCA`).\n")
    f.write("- **SPLIT:** A single TopAneu structure spans multiple distinct P3 categories.\n")
    f.write("- **RELATED:** Anatomically contiguous or branching structures lacking exact boundary correspondence (e.g., TopAneu `R-ICA C7-Pcom-junction` $\\leftrightarrow$ P3 `Right PCom` vs `Right Supraclinoid ICA`).\n")
    f.write("- **NO_DIRECT_EQUIVALENT:** Anatomical variations (e.g., `3rd-A2` median callosal artery) not recognized in P3's standard circle of Willis taxonomy.\n\n")
    f.write("---\n\n")
    f.write("## 2. Aneurysm Location Mapping Table (52 TopAneu Locations)\n\n")
    f.write("| TopAneu ID | TopAneu Location Name | P3 ID | P3 Class Name | Mapping Type | Confidence | Clinical / Anatomical Notes |\n")
    f.write("| :---: | :--- | :---: | :--- | :---: | :---: | :--- |\n")
    for r in mapping_rules:
        if r["category"] == "Location (Aneurysm)":
            f.write(f"| {r['topaneu_id']} | `{r['topaneu_name']}` | {r['p3_id']} | {r['p3_name']} | **{r['mapping_type']}** | {r['confidence']} | {r['notes']} |\n")
    f.write("\n---\n\n")
    f.write("## 3. Vascular Tree Mapping Table (36 TopAneu Vessels)\n\n")
    f.write("| TopAneu ID | TopAneu Vessel Name | P3 ID | P3 Class Name | Mapping Type | Confidence | Clinical / Anatomical Notes |\n")
    f.write("| :---: | :--- | :---: | :--- | :---: | :---: | :--- |\n")
    for r in mapping_rules:
        if r["category"] == "Vessel (Anatomy)":
            f.write(f"| {r['topaneu_id']} | `{r['topaneu_name']}` | {r['p3_id']} | {r['p3_name']} | **{r['mapping_type']}** | {r['confidence']} | {r['notes']} |\n")
print(f"Saved Markdown report to: {MD_PATH}")
