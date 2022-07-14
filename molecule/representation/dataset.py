import logging
from typing import Union
import numpy as np
import pandas as pd

from molecule.prediction.model import FeatureType
from molecule.prediction.feature_extractors import MorganFingerprintFeatureExtractor, SmileFeatureExtractor


class Dataset():
    """
    Represents the dataset
    Contains functions to obtain feature representation etc
    """

    def __init__(self, data: Union[pd.DataFrame, str]) -> None:
        """
        Initialize the dataset either from a pandas dataframe, or from the path to a csv file.
        Assumptions: the dataframe (or the csv) contains at least columns:
            - 'smiles' [str]: SMILE representation of molecule, e.g. 'Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1'
            - 'P1' [int]: whether molecule has property P1: 1 if yes, 0 otherwise
            (if the dataset is used only to make predictions or evaluate a model, 'P1' is not necessary)
        """
        if type(data) is pd.DataFrame:
            self.dataframe = data
        elif type(data) is str:
            self.dataframe = pd.read_csv(data)
        else:
            raise ValueError("'data' must be of type pandas.DataFrame or a path (str) to a csv file.")
        
        # Initialize feature extractors
        self.morgan_feature_extractor = MorganFingerprintFeatureExtractor()
        self.smile_feature_extractor = SmileFeatureExtractor()

    @property
    def size(self) -> int:
        """Return size of dataset"""
        return self.dataframe.shape[0]

    @property
    def labels(self) -> np.ndarray:
        """Return labels: numpy array of ints of shape (self.size,)"""
        return self.dataframe["P1"].to_numpy()

    @property
    def smiles(self) -> np.ndarray:
        """Return SMILE representations: numpy array of strings of shape (self.size,)"""
        return self.dataframe["smiles"].to_numpy()

    def get_features(self, feature_type=FeatureType.MORGAN) -> np.ndarray:
        """
        Return an array of features corresponding the dataset's SMILEs, based on desired feature_type.
        If feature_type is:
            - FeatureType.MORGAN: return Morgan fingerprints representation
            numpy array of shape (self.size, 2048)
            - FeatureType.SMILE: return one-hot-encoding representation of SMILEs
            numpy array of shape (self.size, sequence_length, vocab_size)
        """
        if feature_type == FeatureType.MORGAN:
            return self.morgan_feature_extractor.transform(self.smiles)
        else:
            return self.smile_feature_extractor.transform(self.smiles)