"""
Contains tests related to feature creation.
Contrary to test_features_from_data, does not depend on external files.
"""

import pytest
import pandas as pd
import numpy as np

from molecule.preprocess.feature_extraction import FeatureType
from molecule.preprocess.dataset import Dataset

@pytest.fixture
def example_dataframe():
    """Return a small dataframe for the purpose of tests"""
    df = pd.DataFrame()
    df["P1"] = [1, 1, 0]
    df["mol_id"] = ["CID2106171", "CID1744516", "CID1093792"]
    df["smiles"] = [
        "c1ccc(-c2nnc(CSc3nncs3)o2)cc1",
        "Cc1occc1C(=O)NCCCCNC(=O)c1ccoc1C",
        "Cc1ccc(-c2cc(C(=O)NCCN3CCOCC3)no2)cc1"
    ]
    return df

def test_dataset_creation(example_dataframe):
    """Test that dataset is correctly created from input dataframe"""
    dataset = Dataset(example_dataframe)
    assert dataset.size == 3
    assert (dataset.labels == [1, 1, 0]).all()
    smiles = example_dataframe["smiles"].to_numpy()
    assert (dataset.smiles == smiles).all()

def test_morgan_features(example_dataframe):
    """
    Test that Morgan fingerprints are computed as expected
    Only test on a sample of the fingerprints
    """
    dataset = Dataset(example_dataframe)
    morgan_features = dataset.get_features(FeatureType.MORGAN)
    sample_morgan_features = morgan_features[:,5:10]
    expected_sample_features = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0]
    ])
    assert (sample_morgan_features == expected_sample_features).all()

def test_smile_one_hot_features(example_dataframe):
    """
    Test that one-hot encoded features are computed from SMILE text as expected
    Only test on one molecule on a sample of the full one-hot vector
    """
    dataset = Dataset(example_dataframe)
    smile_features = dataset.get_features(FeatureType.SMILE)
    # Encoding of first word (character in SMILE)
    assert smile_features[0,0,23] == 1.
    assert smile_features[0,0,:].sum() == 1.
    # Encoding of second word
    assert smile_features[0,1,6] == 1.
    assert smile_features[0,1,:].sum() == 1.