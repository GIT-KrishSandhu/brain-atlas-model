import os
import re
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def extract_patient_id(case_id: str) -> str:
    """
    Extract canonical patient ID from TopAneu filename schema:
    Schema: topaneu_{centerID}_{modality}_{patientID}
    Special cases:
      - center4 longitudinal scans (e.g., topaneu_center4_ct_008_1 -> center4_008)
      - center2 multimodal scans (e.g., topaneu_center2_ct_002 & topaneu_center2_mr_002 -> center2_002)
    """
    m = re.match(r"topaneu_([^_]+)_([^_]+)_(.+)", case_id)
    if not m:
        return case_id
    center, mod, pid = m.groups()
    if center == "center4":
        pid = re.sub(r"_\d+$", "", pid)
    return f"{center}_{pid}"


def create_split(
    data_dir: str = "topaneu_release",
    output_csv: str = "experiments/splits/topaneu_v1.csv",
    output_json: str = "experiments/splits/topaneu_v1_summary.json",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    print("=" * 70)
    print("TOPANEU-26 PATIENT-LEVEL STRATIFIED DATA SPLIT CREATION")
    print("=" * 70)
    print(f"Data Directory: {data_dir}")
    print(f"Target Ratios: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
    print(f"Random Seed: {seed}")

    images_dir = os.path.join(data_dir, "images")
    json_dir = os.path.join(data_dir, "location_jsons")

    raw_files = sorted([f for f in os.listdir(images_dir) if f.endswith("_0000.nii.gz") and not f.startswith("._")])
    print(f"Total scan files found: {len(raw_files)}")

    records = []
    for fname in raw_files:
        case_id = fname.replace("_0000.nii.gz", "")
        patient_id = extract_patient_id(case_id)
        
        # Center & Modality
        parts = case_id.split("_")
        center_id = parts[1]
        modality = "MRA" if parts[2] == "mr" else "CTA"

        # Ground truth presence from JSON
        json_path = os.path.join(json_dir, f"{case_id}.json")
        loc_count = 0
        if os.path.exists(json_path):
            with open(json_path, "r") as jf:
                jdata = json.load(jf)
                locs = jdata.get("locations", [])
                loc_count = len(locs)
        presence = 1 if loc_count > 0 else 0

        records.append({
            "case_id": case_id,
            "patient_id": patient_id,
            "center_id": center_id,
            "modality": modality,
            "presence": presence,
            "num_aneurysms": loc_count
        })

    df = pd.DataFrame(records)
    print(f"Constructed case table with {len(df)} entries.")
    
    unique_patients = df["patient_id"].unique()
    print(f"Total unique patients: {len(unique_patients)}")
    assert len(unique_patients) == 408, f"Expected 408 unique patients, found {len(unique_patients)}"

    # Determine patient-level status for stratification
    patient_records = []
    for pid in unique_patients:
        pdf = df[df["patient_id"] == pid]
        p_presence = int(pdf["presence"].max())
        p_center = pdf["center_id"].iloc[0]
        p_modality = pdf["modality"].iloc[0]
        p_scans = len(pdf)
        # Stratification key: presence + modality
        strat_key = f"{p_presence}_{p_modality}"
        patient_records.append({
            "patient_id": pid,
            "patient_presence": p_presence,
            "center_id": p_center,
            "modality": p_modality,
            "num_scans": p_scans,
            "strat_key": strat_key
        })

    pdf_all = pd.DataFrame(patient_records)
    print(f"Patient-level presence distribution: Positive={pdf_all['patient_presence'].sum()}, Negative={(pdf_all['patient_presence'] == 0).sum()}")

    # Stratified Split at patient level
    # 1. Split off test set (15%)
    train_val_patients, test_patients = train_test_split(
        pdf_all,
        test_size=test_ratio,
        random_state=seed,
        stratify=pdf_all["strat_key"]
    )

    # 2. Split train and validation (val_ratio relative to remaining)
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=relative_val_ratio,
        random_state=seed,
        stratify=train_val_patients["strat_key"]
    )

    train_pids = set(train_patients["patient_id"])
    val_pids = set(val_patients["patient_id"])
    test_pids = set(test_patients["patient_id"])

    # Verify no patient overlap
    assert len(train_pids.intersection(val_pids)) == 0, "Patient leakage between train and val!"
    assert len(train_pids.intersection(test_pids)) == 0, "Patient leakage between train and test!"
    assert len(val_pids.intersection(test_pids)) == 0, "Patient leakage between val and test!"

    # Map split to case dataframe
    def assign_split(pid):
        if pid in train_pids:
            return "train"
        elif pid in val_pids:
            return "val"
        elif pid in test_pids:
            return "test"
        raise ValueError(f"Unknown patient {pid}")

    df["split"] = df["patient_id"].apply(assign_split)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved split table to {output_csv}")

    # Summary Statistics
    summary = {
        "dataset_name": "TopAneu-26",
        "total_scans": len(df),
        "total_patients": len(unique_patients),
        "random_seed": seed,
        "splits": {}
    }

    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        unique_p = len(sub["patient_id"].unique())
        pos_cases = int(sub["presence"].sum())
        neg_cases = int((sub["presence"] == 0).sum())
        mra_cases = int((sub["modality"] == "MRA").sum())
        cta_cases = int((sub["modality"] == "CTA").sum())
        pos_patients = int(sub.groupby("patient_id")["presence"].max().sum())
        neg_patients = unique_p - pos_patients

        summary["splits"][s] = {
            "total_cases": len(sub),
            "percentage_cases": round(len(sub) / len(df) * 100, 2),
            "unique_patients": unique_p,
            "percentage_patients": round(unique_p / len(unique_patients) * 100, 2),
            "positive_cases": pos_cases,
            "negative_cases": neg_cases,
            "case_prevalence": round(pos_cases / len(sub), 4),
            "positive_patients": pos_patients,
            "negative_patients": neg_patients,
            "mra_count": mra_cases,
            "cta_count": cta_cases
        }

    with open(output_json, "w") as jf:
        json.dump(summary, jf, indent=2)
    print(f"Saved summary metrics to {output_json}")

    print("\n--- Split Breakdown Summary ---")
    for s, data in summary["splits"].items():
        print(f"[{s.upper():<5}] Cases: {data['total_cases']:>3} ({data['percentage_cases']}%) | Patients: {data['unique_patients']:>3} | Pos/Neg Cases: {data['positive_cases']}/{data['negative_cases']} (Prevalence: {data['case_prevalence']*100:.1f}%) | MRA/CTA: {data['mra_count']}/{data['cta_count']}")

    print("=" * 70)
    print("SPLIT PROTOCOL ESTABLISHED SUCCESSFULLY")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="topaneu_release")
    parser.add_argument("--output_csv", type=str, default="experiments/splits/topaneu_v1.csv")
    parser.add_argument("--output_json", type=str, default="experiments/splits/topaneu_v1_summary.json")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_split(
        data_dir=args.data_dir,
        output_csv=args.output_csv,
        output_json=args.output_json,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
