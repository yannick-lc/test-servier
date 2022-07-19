"""
Contains "high-level" functions to extract features and train a model.
"""

import logging

from molecule.config import configuration
from molecule.preprocess.dataset import Dataset
from molecule.preprocess.feature_extraction import FeatureType
from molecule.train.model import Model


def train_on_dataset(feature_type: FeatureType, path_dataset: str=None, path_to_save_model: str=None) -> Model:
    """
    Train a model on a specified dataset, save the weights in specified path and return the model.
    feature_type is used to determine the type of features (fingerprint or text) and thus the model used.
    If no dataset or save path are specified, default values provided in configuration are used.
    """
    # Load default value if needed
    if path_dataset is None:
        path_dataset = configuration["default_path_to_train_dataset"]
    if path_to_save_model is None:
        path_to_save_model = configuration["default_path_to_model_save"]
    logging.info(f"Training model based on {feature_type} features on {path_dataset}")

    # Train model
    dataset = Dataset(path_dataset)
    model = Model(feature_type)
    X, y = dataset.get_features(feature_type), dataset.labels
    model.fit(X, y, validation_set_ratio=0.)

    # Save and return model
    model.save(path_to_save_model)
    return model