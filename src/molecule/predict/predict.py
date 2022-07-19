"""
Contains "high-level" functions to use a trained model to make predictions,
as well as evaluate the predictions.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, RocCurveDisplay

from molecule.config import configuration
from molecule.preprocess.dataset import Dataset
from molecule.preprocess.feature_extraction import FeatureType
from molecule.train.model import Model



def predict_from_dataset(feature_type: FeatureType, path_dataset: str=None,
            path_to_saved_model: str=None, path_to_save_predictions: str=None) -> np.ndarray:
    """
    Use a trained model to make predictions on the specified dataset,
    and save them in path_to_save_predictions if specified.
    Predictions are probabilities of having property P1 for each sample in the dataset,
    so the returned result is a numpy array of size (n_samples,).
    feature_type is used to determine the type of features (fingerprint or text) and thus the model used.
    If no dataset or saved model path are specified, default values provided in configuration are used.
    """
    # Load default value if needed
    if path_dataset is None:
        path_dataset = configuration["default_path_to_test_dataset"]
    if path_to_saved_model is None:
        path_to_saved_model = configuration["default_path_to_model_save"]
    logging.info(f"Making predictions on dataset {path_dataset} " \
                f"with model {path_to_saved_model} based on {feature_type} features.")

    # Make predictions
    dataset = Dataset(path_dataset)
    model = Model(feature_type)
    model.load(path_to_saved_model)
    X = dataset.get_features(feature_type)
    predicted_proba = model.predict_proba(X)

    # Save predictions
    if path_to_save_predictions is not None:
        dataframe = dataset.dataframe
        dataframe["prediction"] = predicted_proba
        dataframe.to_csv(path_to_save_predictions, index=False)
        logging.info(f"Wrote predictions at {path_to_save_predictions}.")

    return predicted_proba


def evaluate_on_dataset(feature_type: FeatureType, path_dataset: str=None, path_to_saved_model: str=None,
            draw_roc_curve=False) -> float:
    """
    Evaluate the performance of a train model, as measured by the ROC AUCs.
    feature_type is used to determine the type of features (fingerprint or text) and thus the model used.
    If no dataset or saved model path are specified, default values provided in configuration are used.
    Optionnaly, draw the ROC curve.
    """
    # Load default value if needed
    if path_dataset is None:
        path_dataset = configuration["default_path_to_test_dataset"]
    logging.info(f"Evaluating model {path_dataset} " \
                f"with model {path_to_saved_model} based on {feature_type} features.")
    
    # Measure predictions performance
    dataset = Dataset(path_dataset)
    labels = dataset.labels
    predicted_proba = predict_from_dataset(feature_type, path_dataset, path_to_saved_model)
    auc = roc_auc_score(labels, predicted_proba)
    print(f"Area Under Curve is {auc:.3f}")

    # Draw ROC curve
    if draw_roc_curve:
        fig = plt.figure(figsize=(5,5))
        plt.title("ROC curve")
        plt.plot([0,1], [0,1], "--k")
        RocCurveDisplay.from_predictions(labels, predicted_proba, ax=fig.axes[0])
        plt.show()

    return auc