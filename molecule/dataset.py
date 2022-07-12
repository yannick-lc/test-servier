import logging
from enum import Enum
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from molecule.predictors.feature_extractors import (bag_of_words_encoder,
            smiles_to_one_hot_array_features, fingerprint_features)


class FeatureType(Enum):
    """
    Represents choices for type of features (and therefore model architecture)
    """
    MORGAN = 1
    SMILE = 2


class Dataset():
    """
    Represents the dataset
    Contains functions to obtain feature representation etc
    """

    def __init__(self, dataframe: pd.DataFrame, encoder: OneHotEncoder=None) -> None:
        """
        Initialize the dataset from provided data.
        Assumptions: dataframe contains at least columns:
            - 'smiles' [str]: SMILE representation of molecule, e.g. 'Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1'
            - 'P1' [int]: whether molecule has property P1: 1 if yes, 0 otherwise
        
        The encoder is used to convert SMILE characters to bag-of-words representations.
        It can be provided if the objective is to evaluate a trained model.
        Otherwise, it will be fitted to the data if necessary.
        """
        self.dataframe = dataframe
        self.encoder = encoder

    @property
    def size(self) -> int:
        """Return size of dataset"""
        return self.dataframe.shape[0]

    @property
    def vocab_size(self) -> int:
        """Return size of vocabulary, i.e. number of distinct characters in SMILEs representations"""
        if self.encoder is None:
            logging.info(
                'No encoder found. ' \
                'Fitting bag-of-words model to provided data and saving it for future use.'
            )
            self.encoder = bag_of_words_encoder(self.smiles)
        vocab_size = self.encoder.categories_[0].shape[0]
        return vocab_size

    @property
    def labels(self) -> np.ndarray:
        """Return labels: numpy array of ints of shape (self.size,)"""
        return self.dataframe['P1'].to_numpy()

    @property
    def smiles(self) -> np.ndarray:
        """Return SMILE representations: numpy array of strings of shape (self.size,)"""
        return self.dataframe['smiles'].to_numpy()

    def get_features(self, feature_type=FeatureType.MORGAN) -> np.ndarray:
        """
        Return
        """
        if feature_type == FeatureType.MORGAN:
            return self._get_features_morgan_fingerprint()
        else:
            return self._get_features_bag_of_words()

    def _get_features_morgan_fingerprint(self) -> np.ndarray:
        """
        Return Morgan fingerprints representation of SMILEs:
        numpy array of shape (self.size, 2048)
        """
        fingerprints = np.array([fingerprint_features(smile).ToList() for smile in self.smiles])
        return fingerprints


    def _get_features_bag_of_words(self) -> np.ndarray:
        """
        Return bag-of-words representation of SMILEs:
        numpy array of shape (self.size, sequence_length, vocab_size)
        """
        if self.encoder is None:
            logging.info(
                'No encoder found. ' \
                'Fitting bag-of-words model to provided data and saving it for future use.'
            )
            self.encoder = bag_of_words_encoder(self.smiles)
        else:
            logging.info('Using existing bag-of-words encoder.')
        bag_of_words = smiles_to_one_hot_array_features(self.smiles, self.encoder)
        return bag_of_words