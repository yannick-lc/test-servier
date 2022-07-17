"""
Contains tests related to feature creation
Depends on files located in test_data
"""

import pytest
import os
import numpy as np

from molecule.prediction.model import FeatureType
from molecule.representation.dataset import Dataset
from molecule.config import get_project_root_directory

def absolute_path(relative_path: str):
    """Return absolute path from relative path based on project root"""
    path_to_root = get_project_root_directory()
    absolute_path = os.path.join(path_to_root, relative_path)
    return absolute_path

@pytest.fixture
def sample_dataset():
    """Return a dataset for the purpose of tests"""
    RELATIVE_PATH_TO_DATASET = "tests/test_data/dataset.csv"
    path_to_dataset = absolute_path(RELATIVE_PATH_TO_DATASET)
    dataset = Dataset(path_to_dataset)
    return dataset

@pytest.fixture
def sample_labels():
    """Return some labels for the purpose of tests"""
    RELATIVE_PATH_TO_LABELS = "tests/test_data/labels.npy"
    path_to_labels = absolute_path(RELATIVE_PATH_TO_LABELS)
    labels = np.load(path_to_labels)
    return labels

@pytest.fixture
def sample_morgan_features():
    """Return Morgan fingerprints features for the purpose of tests"""
    RELATIVE_PATH_TO_FEATURES = "tests/test_data/features_morgan.npy"
    path_to_features = absolute_path(RELATIVE_PATH_TO_FEATURES)
    features = np.load(path_to_features)
    return features

@pytest.fixture
def sample_smile_features():
    """Return SMILE one-hot features for the purpose of tests"""
    RELATIVE_PATH_TO_FEATURES = "tests/test_data/features_smile.npy"
    path_to_features = absolute_path(RELATIVE_PATH_TO_FEATURES)
    features = np.load(path_to_features)
    return features

def test_dataset_creation(sample_dataset, sample_labels):
    """Test that dataset has been correctly created"""
    assert sample_dataset.size == 100
    assert (sample_dataset.labels == sample_labels).all()

def test_morgan_features(sample_dataset, sample_morgan_features):
    """Test that Morgan fingerprints are computed as expected"""
    morgan_features = sample_dataset.get_features(FeatureType.MORGAN)
    assert (morgan_features == sample_morgan_features).all()

def test_smile_one_hot_features(sample_dataset, sample_smile_features):
    """Test that one-hot encoded features are computed from SMILE text as expected"""
    smile_features = sample_dataset.get_features(FeatureType.SMILE)
    assert (smile_features == sample_smile_features).all()