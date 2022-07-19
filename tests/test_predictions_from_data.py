"""
Contains tests related to using a trained model to make predictions
Depends on files located in tests/test_data/
Also depends on pre-trained models localed in models/

IMPORTANT WARNING: this is a test for functionalities only:
predictions are done on data that may have been seen by the model during training.
As such, PERFORMANCE IS NOT REPRESENTATIVE.
"""

import pytest
import numpy as np

from molecule.preprocess.feature_extraction import FeatureType
from molecule.train.model import Model
from test_features_from_data import absolute_path, sample_morgan_features, sample_smile_features

@pytest.fixture
def sample_model_morgan():
    """Return pre-trained model based on Morgan fingerprints"""
    model = Model(FeatureType.MORGAN)
    RELATIVE_PATH_TO_MODEL = "models/model_morgan.pth"
    path_to_model = absolute_path(RELATIVE_PATH_TO_MODEL)
    model.load(path_to_model)
    return model

@pytest.fixture
def sample_model_smile():
    """Return pre-trained model based on SMILE one-hot encoding"""
    model = Model(FeatureType.SMILE)
    RELATIVE_PATH_TO_MODEL = "models/model_smile.pth"
    path_to_model = absolute_path(RELATIVE_PATH_TO_MODEL)
    model.load(path_to_model)
    return model

@pytest.fixture
def sample_predictions_morgan():
    """Return predictions of pre-trained model based on Morgan features"""
    RELATIVE_PATH_TO_PREDICTIONS = "tests/test_data/predictions_morgan.npy"
    path_to_predictions = absolute_path(RELATIVE_PATH_TO_PREDICTIONS)
    predictions = np.load(path_to_predictions)
    return predictions

@pytest.fixture
def sample_predictions_smile():
    """Return predictions of pre-trained model based on SMILE features"""
    RELATIVE_PATH_TO_PREDICTIONS = "tests/test_data/predictions_smile.npy"
    path_to_predictions = absolute_path(RELATIVE_PATH_TO_PREDICTIONS)
    predictions = np.load(path_to_predictions)
    return predictions

def test_prediction_morgan(sample_morgan_features, sample_model_morgan, sample_predictions_morgan):
    """Test that pre-trained model predictions correspond to saved predictions"""
    predicted_proba = sample_model_morgan.predict_proba(sample_morgan_features)
    assert (predicted_proba == sample_predictions_morgan).all()

def test_prediction_smile(sample_smile_features, sample_model_smile, sample_predictions_smile):
    """Test that pre-trained model predictions correspond to saved predictions"""
    predicted_proba = sample_model_smile.predict_proba(sample_smile_features)
    assert (predicted_proba == sample_predictions_smile).all()