from typing import List, Union

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from args import TrainArgs
from features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph, \
    BatchMolGraphWithSubstructures, get_atom_fdim_with_substructures, \
    mol2graph_with_substructures
from nn_utils import index_select_ND, get_activation_function


class SubstructureEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, atom_fdim: int):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(SubstructureEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.args = args
        self.hidden_size = args.substructures_hidden_size
        self.bias = args.bias
        self.dropout = args.dropout
        self.use_input_features = args.use_input_features
        self.device = args.device

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        self.W_o = nn.Linear(self.atom_fdim, self.hidden_size)

    def forward(self,
                mol_graph: Union[BatchMolGraphWithSubstructures],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.
        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :param features_batch: A list of numpy arrays containing additional features.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """

        f_atoms, a_scope = mol_graph.get_components()
        f_atoms = f_atoms.to(self.device)

        a_input = f_atoms
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class SubstructureLayer(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        self.args = args
        super(SubstructureLayer, self).__init__()
        self.atom_fdim = get_atom_fdim_with_substructures(use_substructures=args.substructures_use_substructures,
                                                          merge_cycles=args.substructures_merge)
        self.encoder = SubstructureEncoder(args, self.atom_fdim)

    def forward(self,
                batch: Union[List[str], List[Chem.Mol], BatchMolGraphWithSubstructures],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecules.
        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch: A list of numpy arrays containing additional features.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if type(batch) != BatchMolGraphWithSubstructures:
            batch = mol2graph_with_substructures(batch, args=self.args)

        output = self.encoder.forward(batch, features_batch)

        return output