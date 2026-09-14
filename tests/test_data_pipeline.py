import os
import sys
import torch
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.dataset import TopAneuDataset


def test_topaneu_dataset_item():
    split_csv = "experiments/splits/topaneu_v1.csv"
    assert os.path.exists(split_csv)
    df = pd.read_csv(split_csv).head(2)

    dataset = TopAneuDataset(
        split_df=df,
        data_dir="topaneu_release",
        cache_dir="scratch/cache_224",
        target_size=(224, 224, 224)
    )

    assert len(dataset) == 2
    item = dataset[0]

    assert "case_id" in item
    assert "image" in item
    assert "presence" in item
    assert "modality" in item

    assert item["image"].shape == (1, 224, 224, 224)
    assert item["presence"].shape == (1,)
    assert item["presence"].dtype == torch.float32
    assert item["modality"].dtype == torch.long
