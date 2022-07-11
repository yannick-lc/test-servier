import logging
from typing import List
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def get_bag_of_words_encoder(smiles: List[str]) -> OneHotEncoder:
    """
    Fit bag-of-words model on list of SMILE representations using scikit-learn one-hot encoder
    """
    # Assessing vocabulary (distinct characters in SMILE representations)
    vocab = set().union(*[set(smile) for smile in smiles]) # distinct characters
    logging.info(f'Found {len(vocab)} distinct characters.')

    # Fitting bag-of-words encoder (one-hot encoder)
    vocab_array = np.array(list(vocab)).reshape(-1,1)
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoder.fit(vocab_array)
    return encoder


def smiles_to_one_hot_array(smiles: List[str], encoder: OneHotEncoder, max_length=None) -> np.array:
    """
    Convert list of smiles (str) to numpy array provided bag-of-words encoder.
    Arrays are padded so that all sequences have a length of size max_length.
    If max_length is not provided, it is set to the length of the longest sequence.
    Output size: (n_smiles, sequence_length, vocab_size)
    """
    if max_length is None:
        max_length = max([len(s) for s in smiles])
    vocab_size = encoder.categories_[0].shape[0]
    
    arrays = []
    for smile in smiles:
        # One hot encoding
        chars_array = np.array(list(smile)).reshape(-1,1)
        one_hot_array = encoder.transform(chars_array)
        length = one_hot_array.shape[0]
        
        if length > max_length:
            raise ValueError(f'Found sequence of length {length} ' \
                f'while max sequence length is {max_length}. ' \
                'Please increase max_length.')
        
        # Padding with zero to reach specified sequence length
        zero_padding = np.zeros((max_length - length, vocab_size))
        padded_array = np.concatenate([one_hot_array, zero_padding])
        arrays.append(padded_array)
        
    result = np.stack(arrays)
    return result