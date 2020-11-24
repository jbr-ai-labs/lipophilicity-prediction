from typing import List, Tuple, Union

import torch


from .molecule import create_molecule_for_smiles

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))


def get_atom_fdim_with_substructures(use_substructures=False, merge_cycles=False) -> int:
    """Gets the dimensionality of the atom feature vector."""
    atom_fdim = 160
    if use_substructures:
        atom_fdim += 5
    if merge_cycles:
        atom_fdim += 5
    return atom_fdim


def atom_features_for_substructures(atom) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.
    :param atom: An RDKit atom.
    :return: A list containing the atom features.
    """
    return atom.get_representation()


class MolGraphWithSubstructures:
    """
    A :class:`MolGraph` represents the graph structure and featurization of a single molecule.
    A MolGraph computes the following attributes:
    * :code:`n_atoms`: The number of atoms in the molecule.
    * :code:`n_bonds`: The number of bonds in the molecule.
    * :code:`f_atoms`: A mapping from an atom index to a list of atom features.
    """

    def __init__(self, mol: str, args):
        """
        :param mol: A SMILES or an RDKit molecule.
        """
        mol = create_molecule_for_smiles(mol, args)

        self.n_atoms = 0  # number of atoms
        self.f_atoms = []  # mapping from atom index to atom features

        # Get atom features
        self.f_atoms = [atom_features_for_substructures(atom) for atom in mol.get_atoms()]
        self.n_atoms = len(self.f_atoms)


class BatchMolGraphWithSubstructures:
    """
    A :class:`BatchMolGraphWithSubstructures` represents the graph structure and featurization of a batch of molecules.
    A BatchMolGraph contains the attributes of a :class:`MolGraphWithSubstructures` plus:
    * :code:`atom_fdim`: The dimensionality of the atom feature vector.
    * :code:`a_scope`: A list of tuples indicating the start and end atom indices for each molecule.
    * :code:`a2a`: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraphWithSubstructures], args):
        r"""
        :param mol_graphs: A list of :class:`MolGraphWithSubstructures`\ s from which to construct the
        :class:`BatchMolGraphWithSubstructures`.
        """
        self.atom_fdim = get_atom_fdim_with_substructures(use_substructures=args.substructures_use_substructures,
                                                          merge_cycles=args.substructures_merge)

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.n_atoms += mol_graph.n_atoms

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.a2a = None  # only needed if using atom messages

    def get_components(self) -> Tuple[torch.FloatTensor, List[Tuple[int, int]]]:
        """
        Returns the components of the :class:`BatchMolGraphWithSubstructures`.
        The returned components are, in order:
        * :code:`f_atoms`
        * :code:`a_scope`
        :return: A tuple containing PyTorch tensors with the atom features, graph structure,
                 and scope of the atoms (i.e., the indices of the molecules they belong to).
        """
        return self.f_atoms, self.a_scope

    def get_a2a(self) -> torch.LongTensor:
        """
        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        return self.a2a


def mol2graph_with_substructures(mols: Union[List[str]], args) -> BatchMolGraphWithSubstructures:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraphWithSubstructures` containing the batch of
    molecular graphs.
    :param mols: A list of SMILES or a list of RDKit molecules.
    :return: A :class:`BatchMolGraphWithSubstructures` containing the combined molecular graph for the molecules.
    """
    return BatchMolGraphWithSubstructures(
        [MolGraphWithSubstructures(mol, args) for mol in mols], args=args)