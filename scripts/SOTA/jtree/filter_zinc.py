import pandas as pd
import os

import sys
sys.path.append('../../../../../icml18-jtnn')
sys.path.append('../../../../../icml18-jtnn/jtnn')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser
from collections import deque

# from jtnn import *
import rdkit

from jtnn_enc import JTNNEncoder

VOCAB_PATH = '../../../../../icml18-jtnn/data/zinc_our/vocab.txt'
DATA_PATH = '../../../../../icml18-jtnn/data/zinc_our'

RAW_PATH = '../../../data/raw/baselines/jtree'

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

import math, random, sys
from optparse import OptionParser
from collections import deque

import rdkit
import rdkit.Chem as Chem

from mol_tree import Vocab, MolTree
from jtnn_vae import JTNNVAE
from jtprop_vae import JTPropVAE
from mpn import MPN, mol2graph
from nnutils import create_var
from datautils import MoleculeDataset, PropDataset
from chemutils import decode_stereo

from tqdm import tqdm
vocab = [x.strip("\r\n ") for x in open(VOCAB_PATH)] 
vocab = Vocab(vocab)

hidden_size = int(450)
latent_size = int(56)
depth = int(3)
stereo = True if int(1) == 1 else False

model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)
model = model.cuda()

SMILES = [x.strip("\r\n ") for x in open(os.path.join(DATA_PATH, 'all.txt'))] 

import numpy as np    
broken_smiles = {}
ok_smiles = []
for smiles in tqdm(SMILES):
    try:
        latent_representation = model.encode_latent_mean([smiles])
    except (KeyError, RuntimeError), e:
        broken_smiles[smiles]=e
        continue
    ok_smiles.append(smiles)
print(len(ok_smiles))
with open(os.path.join(DATA_PATH, 'all_filtered.txt'),'w') as f:
    f.write('\n'.join(ok_smiles))