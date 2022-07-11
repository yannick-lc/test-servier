from typing import List
from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

def fingerprint_features(smile_string: List[str], radius: int=2, size: int=2048) -> ExplicitBitVect:
    """
    Provided function.
    Return Morgan fingerprint of provided SMILE string in the form of an ExplicitBitVect
    """
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    bitvect = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius,
        nBits=size,
        useChirality=True,
        useBondTypes=True,
        useFeatures=False
    )
    return bitvect