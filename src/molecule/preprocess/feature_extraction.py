"""
Contains classes and utilities to extract features from data
"""

from typing import List
from enum import Enum

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from molecule.config import configuration


class FeatureType(Enum):
    """Represents choices for type of features (and therefore model architecture)"""
    MORGAN = 1
    SMILE = 2


class SmileFeatureExtractor:
    """
    Used to vectorize SMILE text representations to character-based one-hot encoded vectors
    """
    def __init__(self) -> None:
        """
        Load vocabulary (possible characters in SMILEs) from json config file
        and fit scikit-learn OneHotEncoder
        """
        self.vocabulary = np.array(list(configuration["vocabulary"]))
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.encoder.fit(self.vocabulary.reshape(-1,1))

    @property
    def vocabulary_size(self) -> int:
        """Return umber of words in vocabulary, i.e. characters in SMILEs"""
        return self.vocabulary.shape[0]

    def encode_smile(self, smile: str, target_length: int=None) -> np.ndarray:
        """
        Convert a single smile (str) to a numpy array using one-hot encoding.
        Array is padded to have a length of target_length.
        Output size: (n_smiles, sequence_length, vocab_size)
        """
        # Check that target length is consistent with input
        chars_array = np.array(list(smile))
        length = len(chars_array)
        if length > target_length:
            raise ValueError(f'Found sequence of length {length} ' \
                f'while target sequence length is {target_length}. ' \
                'Please increase or remove target_length.')

        # Perform one-hot encoding
        one_hot_array = self.encoder.transform(chars_array.reshape(-1,1))

        # Pad with zero to reach target sequence length
        zero_padding = np.zeros((target_length - length, self.vocabulary_size))
        padded_array = np.concatenate([one_hot_array, zero_padding])
        return padded_array

    def transform(self, smiles: List[str], max_length: int=None) -> np.ndarray:
        """
        Convert a list of smiles (str) to a numpy array using one-hot encoding.
        Arrays are padded so that all sequences have a length of max_length.
        If max_length is not provided, it is set to the length of the longest sequence.
        Output size: (n_smiles, sequence_length, vocab_size)
        """
        # Set max_length to length of longest string if not provided
        if max_length is None:
            max_length = max([len(s) for s in smiles])

        # Convert all strings to sequences of one-hot encoded vectors of same length
        one_hot_arrays = []
        for smile in smiles:
            one_hot_array = self.encode_smile(smile, target_length=max_length)
            one_hot_arrays.append(one_hot_array)
        result = np.stack(one_hot_arrays)
        return result


class MorganFingerprintFeatureExtractor:
    """
    Used to extract Morgan fingerprints to SMILE text representations
    """
    def __init__(self, radius=2, fingerprint_size=2048) -> None:
        """Initialize with desired properties"""
        self.radius = radius
        self.fingerprint_size = fingerprint_size

    def extract_fingerprint(self, smile: str) -> ExplicitBitVect:
        """
        Based on provided fingerprint_features function:
        return Morgan fingerprint of provided SMILE string in the form of an ExplicitBitVect
        """
        mol = MolFromSmiles(smile)
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
        bitvect = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
            self.radius, nBits=self.fingerprint_size,
            useChirality=True, useBondTypes=True, useFeatures=False)
        return bitvect

    def transform(self, smiles: List[str]) -> np.ndarray:
        """
        Convert a list of smiles (str) to a matrix of Morgan fingerprint representations
        Output size: (n_smiles, fingerprint_size)
        """
        fingerprint_arrays = []
        for smile in smiles:
            fingerprint_array = self.extract_fingerprint(smile)
            fingerprint_arrays.append(fingerprint_array)
        result = np.stack(fingerprint_arrays)
        return result
