from typing import Callable, List, Union

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


Molecule = Union[str, Chem.Mol]
FeaturesGenerator = Callable[[Molecule], np.ndarray]


FEATURES_GENERATOR_REGISTRY = {}


def register_features_generator(features_generator_name: str) -> Callable[[FeaturesGenerator], FeaturesGenerator]:
    """
    Creates a decorator which registers a features generator in a global dictionary to enable access by name.

    :param features_generator_name: The name to use to access the features generator.
    :return: A decorator which will add a features generator to the registry using the specified name.
    """
    def decorator(features_generator: FeaturesGenerator) -> FeaturesGenerator:
        FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator

    return decorator


def get_features_generator(features_generator_name: str) -> FeaturesGenerator:
    """
    Gets a registered features generator by name.

    :param features_generator_name: The name of the features generator.
    :return: The desired features generator.
    """
    if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found. '
                         f'If this generator relies on rdkit features, you may need to install descriptastorus.')

    return FEATURES_GENERATOR_REGISTRY[features_generator_name]


def get_available_features_generators() -> List[str]:
    """Returns a list of names of available features generators."""
    return list(FEATURES_GENERATOR_REGISTRY.keys())


MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048


@register_features_generator('morgan')
def morgan_binary_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    Generates a binary Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the binary Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


@register_features_generator('morgan_count')
def morgan_counts_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    Generates a counts-based Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the counts-based Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


@register_features_generator('rdkit_2d')
def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
    """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
    raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
                      '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D features.')
    
@register_features_generator('rdkit_2d_best')
def rdkit_2d_features_generator_best(mol: Molecule) -> np.ndarray:
    """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
    raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
                      '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D features.')


@register_features_generator('rdkit_2d_normalized')
def rdkit_2d_normalized_features_generator(mol: Molecule) -> np.ndarray:
    """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
    raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
                      '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D normalized features.')
    
@register_features_generator('rdkit_2d_normalized_best')
def rdkit_2d_normalized_features_generator_best(mol: Molecule) -> np.ndarray:
    
    """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
    raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
                      '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D normalized features.')
    
@register_features_generator('rdkit_2d_normalized_wo_MolLogP')
def rdkit_2d_normalized_wo_MolLogP(mol: Molecule) -> np.ndarray:
    
    """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
    raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
                      '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D normalized features.')
    
@register_features_generator('rdkit_wo_fragments_and_counts')
def rdkit_wo_fragments_and_counts(mol: Molecule) -> np.ndarray:
    
    """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
    raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
                      '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D normalized features.')


try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

    @register_features_generator('rdkit_2d')
    def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D features for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdDescriptors.RDKit2D()
        features = generator.process(smiles)[1:]
        features = np.array(features).astype(float)
        features = np.where(np.isnan(features), np.zeros(features.shape), features)
        features = np.where(np.isinf(features), np.zeros(features.shape), features)
        return features
    
    @register_features_generator('rdkit_2d_best')
    def rdkit_2d_features_generator_best(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D features for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D features.
        """
        import os
        RAW_PATH = './data/raw/baselines/dmpnn'
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdDescriptors.RDKit2D()
        features = generator.process(smiles)[1:]
        
        feature_names = [feature[0] for feature in generator.columns]

        feature_dict = dict(zip(feature_names, features))
        
        with open(os.path.join(RAW_PATH,'RDKitBestfeatures.txt'),'r') as f:
            best_feature_names = f.read().split('\n')
            
        best_features = []
        for best_feature_name in best_feature_names:

            best_features.append(feature_dict[best_feature_name])

        best_features = np.array(best_features).astype(float)
        best_features = np.where(np.isnan(best_features), np.zeros(best_features.shape), best_features)
        best_features = np.where(np.isinf(best_features), np.zeros(best_features.shape), best_features)
        return best_features


    @register_features_generator('rdkit_2d_normalized')
    def rdkit_2d_normalized_features_generator(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D normalized features for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D normalized features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]

        return features
    

    @register_features_generator('rdkit_2d_normalized_best')
    def rdkit_2d_normalized_features_generator_best(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D normalized features for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D normalized features.
        """
        import os
        RAW_PATH = './data/raw/baselines/dmpnn'
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]
        
        feature_names = [feature[0] for feature in generator.columns]

        feature_dict = dict(zip(feature_names, features))
        
        with open(os.path.join(RAW_PATH,'RDKitBestfeatures.txt'),'r') as f:
            best_feature_names = f.read().split('\n')
            
        best_features = []
        for best_feature_name in best_feature_names:

            best_features.append(feature_dict[best_feature_name])

        return best_features
    
    @register_features_generator('rdkit_2d_normalized_wo_MolLogP')
    def rdkit_2d_normalized_wo_MolLogP(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D normalized features for a molecule without features corresponding the count of
        specific fragments in a molecule or count of fragment types (i.e. ring count).
        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit features without fragment features.
        """
        feature_names = rdDescriptors.RDKIT_PROPS["1.0.0"].copy()
        feature_names_copy = feature_names.copy()
        for name in feature_names:
            if name == 'MolLogP':
                feature_names_copy.remove(name)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdNormalizedDescriptors.RDKit2DNormalized(feature_names_copy)
        features = generator.process(smiles)[1:]
        return features
    
    @register_features_generator('rdkit_wo_fragments_and_counts')
    def rdkit_wo_fragments_and_counts(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D normalized features for a molecule without features corresponding the count of
        specific fragments in a molecule or count of fragment types (i.e. ring count).
        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit features without fragment features.
        """
        feature_names = rdDescriptors.RDKIT_PROPS["1.0.0"].copy()
        feature_names_copy = feature_names.copy()
        for name in feature_names:
            if name.startswith("fr") or "count" in name.lower() or "num" in name.lower() or name == 'MolLogP':
                feature_names_copy.remove(name)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdNormalizedDescriptors.RDKit2DNormalized(feature_names_copy)
        features = generator.process(smiles)[1:]
        return features
    
except ImportError:
    pass



"""
Custom features generator template.

Note: The name you use to register the features generator is the name
you will specify on the command line when using the --features_generator <name> flag.
Ex. python train.py ... --features_generator custom ...

@register_features_generator('custom')
def custom_features_generator(mol: Molecule) -> np.ndarray:
    # If you want to use the SMILES string
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

    # If you want to use the RDKit molecule
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    # Replace this with code which generates features from the molecule
    features = np.array([0, 0, 1])

    return features
"""
