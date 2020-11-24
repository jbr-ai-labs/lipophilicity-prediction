import argparse


import shutil

import sys
sys.path.append('../../../../../icml18-jtnn')
sys.path.append('../../../../../icml18-jtnn/jtnn')

import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser
from collections import deque

# from jtnn import *
import rdkit

# from jtnn_enc import JTNNEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

import pandas as pd

import os
import shutil

import json
import pickle

from tensorboardX import SummaryWriter

criterion = nn.MSELoss()

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1
            
from torch.utils.data import Dataset
from mol_tree import MolTree, Vocab
import numpy as np
from jtnn_enc import JTNNEncoder
from mpn import MPN,  mol2graph

SMILES_TO_MOLTREE = {}

class MoleculeDataset(Dataset):

    def __init__(self, data_file, args, SMILES_COLUMN = 'smiles', TARGET_COLUMN = 'logP'):
        global SMILES_TO_MOLTREE
        
        self.data = pd.read_csv(data_file)
        data_options = ['train','val','test']
        
        for option in data_options:
            if option in data_file:
                broken_smiles  = [x.strip("\r\n ") for x in open(os.path.join(args.raw_path,option+'_errs.txt'))] 
                
        self.data = self.data[~self.data[SMILES_COLUMN].isin(broken_smiles)]
        self.SMILES_COLUMN = SMILES_COLUMN
        self.TARGET_COLUMN = TARGET_COLUMN
        
        for i in tqdm(range(len(self.data))):
            if self.data.iloc[i][SMILES_COLUMN] in SMILES_TO_MOLTREE:
                mol_tree = SMILES_TO_MOLTREE[self.data.iloc[i][SMILES_COLUMN]]
            else:
                mol_tree = MolTree(self.data.iloc[i][SMILES_COLUMN])
                SMILES_TO_MOLTREE[self.data.iloc[i][SMILES_COLUMN]] = mol_tree
                mol_tree.recover()
                mol_tree.assemble()
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        global SMILES_TO_MOLTREE
        smiles = self.data.iloc[idx][self.SMILES_COLUMN]
        target = self.data.iloc[idx][self.TARGET_COLUMN]
        if smiles in SMILES_TO_MOLTREE.keys():
            mol_tree = SMILES_TO_MOLTREE[smiles]
        else:
            mol_tree = MolTree(smiles)
            SMILES_TO_MOLTREE[smiles] = mol_tree
        return mol_tree, target            
            
class JTPredict(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, depth, stereo=True):
        super(JTPredict, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth

        self.embedding = nn.Embedding(vocab.size(), hidden_size)
        self.jtnn = JTNNEncoder(vocab, hidden_size, self.embedding)
        self.mpn = MPN(hidden_size, depth)
        
        self.output_size = 1

        self.T_mean = nn.Linear(hidden_size, latent_size / 2)
        self.T_var = nn.Linear(hidden_size, latent_size / 2)
        self.G_mean = nn.Linear(hidden_size, latent_size / 2)
        self.G_var = nn.Linear(hidden_size, latent_size / 2)
        
        self.use_stereo = stereo
        if stereo:
            self.stereo_loss = nn.CrossEntropyLoss(size_average=False)
    
    def encode(self, mol_batch):
        set_batch_nodeID(mol_batch, self.vocab)
        root_batch = [mol_tree.nodes[0] for mol_tree in mol_batch]
        tree_mess,tree_vec = self.jtnn(root_batch)

        smiles_batch = [mol_tree.smiles for mol_tree in mol_batch]
        mol_vec = self.mpn(mol2graph(smiles_batch))
        return tree_mess, tree_vec, mol_vec

    def encode_latent_mean(self, smiles_list):
        print(smiles_list)
        mol_batch = [MolTree(s) for s in smiles_list]
#         print(mol_batch)
        for mol_tree in mol_batch:
            mol_tree.recover()

        _, tree_vec, mol_vec = self.encode(mol_batch)
        tree_mean = self.T_mean(tree_vec)
        mol_mean = self.G_mean(mol_vec)
        return torch.cat([tree_mean,mol_mean], dim=1)
    
    def create_ffn(self, ffn_num_layers = 3, ffn_hidden_size = 50):
        """
        Creates the feed-forward layers for the model.
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        dropout = nn.Dropout(0.5)
        activation = nn.ReLU()
        
        first_linear_dim = self.latent_size

        # Create FFN layers
        if ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, ffn_hidden_size)
            ]
            for _ in range(ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(ffn_hidden_size, ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(ffn_hidden_size, self.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, mol_batch, beta=0):
        batch_size = len(mol_batch)

        _, tree_vec, mol_vec = self.encode(mol_batch)
        
        tree_mean = self.T_mean(tree_vec)
        mol_mean = self.G_mean(mol_vec)
        
        feature_vec =  torch.cat([tree_mean, mol_mean], dim=1)
        
        return self.ffn(feature_vec)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.epoch = 0

    def __call__(self, val_loss, model, epoch = 0):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.epoch = epoch
            print 'EarlyStopping counter: ',self.counter,  ' out of ', self.patience
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print 'Validation loss decreased', round(self.val_loss_min,6),' -->', round(val_loss,6), '.  Saving model ...'
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        X = []
        y = []
        for elem in batch:
            X.append(elem[0])
            y.append(elem[1])
        y = torch.Tensor(y).to(device)
        pred = model(X)
        y = y.view(pred.shape).to(torch.float64)


        loss = criterion(pred.double(), y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


def eval(args, model, device, loader, scaler, train = False, test = False):
    model.eval()
    y_true = []
    y_scores = []
    X_smiles = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        X = []
        y = []
        for elem in batch:
            X.append(elem[0])
            y.append(elem[1])
        y = torch.Tensor(y).to(device)

        with torch.no_grad():
            pred = model(X)

        y_true.append(y.view(pred.shape))
        X_smiles = X_smiles+[x.smiles for x in X]
        if train:
            y_scores.append(pred)
        else:
            y_scores.append(torch.Tensor(scaler.inverse_transform(pred.cpu())))

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    
    if test:
        test_predictions = pd.DataFrame(columns = ['smiles', 'logP', 'logP_pred'])
        test_predictions['smiles'] = X_smiles
        test_predictions['logP'] =y_true
        test_predictions['logP_pred'] = y_scores
        test_predictions.to_csv(os.path.join(args.fname, 'test_predictions.csv'))
    else:
        pass
        

    r2 = r2_score(y_true, y_scores)
    rmse = mean_squared_error(y_true, y_scores)**0.5

    return r2, rmse




def main():
    global SMILES_TO_MOLTREE
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')

    parser.add_argument('--decay', type=float, default=0.005,
                        help='weight decay (default: 0.005)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 3).')
    parser.add_argument('--emb_dim', type=int, default=56,
                        help='embedding dimensions (default: 56)')
    parser.add_argument('--hidden_size', type=float, default=450,
                        help='hidden size of ffn (default: 450)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience of early stopping (default: 50)')
    parser.add_argument('--raw_path', type=str, default='../../../data/raw/baselines/jtree/',
                        help='path to broken smiles')
    parser.add_argument('--dataset', type=str, default = '../../../data/3_final_data/split_data', help='root directory of dataset. For now, only classification.')
    
    parser.add_argument('--vocab_path', type=str, default = '../../../../icml18-jtnn/data/zinc/vocab.txt', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = 'exp_1', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--log_path', type=str, default='../../../data/raw/baselines/jtree/logs', help='root directory for logs')
    parser.add_argument('--model_name', type=str, default= 'MPNVAE-h450-L56-d3-beta0.005/model.iter-4', help='root directory for logs')
    args = parser.parse_args()


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
        
    vocab = [x.strip("\r\n ") for x in open(args.vocab_path)] 
    vocab = Vocab(vocab)
    
    if os.path.exists(os.path.join(args.raw_path, 'SMILES_TO_MOLTREE.pickle')):
        with open(os.path.join(args.raw_path, 'SMILES_TO_MOLTREE.pickle'), 'rb') as handle:
            SMILES_TO_MOLTREE = pickle.load(handle)
        print ('Preprocesed molecules have been loaded')

    batch_size = args.batch_size
    hidden_size = args.hidden_size
    latent_size = args.emb_dim
    depth = args.num_layer
    beta = args.decay
    lr = args.lr
    stereo = True #if int(1) == 1 else False
    
    
    #set up dataset
    train_dataset = MoleculeDataset(os.path.join(args.dataset, 'logp_wo_averaging_train.csv'), args)
    test_dataset = MoleculeDataset(os.path.join(args.dataset, 'logp_wo_averaging_test.csv'), args)
    valid_dataset = MoleculeDataset(os.path.join(args.dataset, 'logp_wo_averaging_validation.csv'), args)
    
    if not os.path.exists(os.path.join(args.raw_path, 'SMILES_TO_MOLTREE.pickle')):
        with open(os.path.join(args.raw_path, 'SMILES_TO_MOLTREE.pickle'), 'wb') as handle:
              pickle.dump(SMILES_TO_MOLTREE, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(len(train_dataset))
    print(len(valid_dataset))
    print(len(test_dataset))

    scaler = StandardScaler()
    scaled_y = torch.tensor(scaler.fit_transform(train_dataset.data[train_dataset.TARGET_COLUMN].values.reshape(-1, 1)).reshape(-1))
    train_dataset.data[train_dataset.TARGET_COLUMN] = scaled_y



    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x, num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x, num_workers=args.num_workers, drop_last=True)

    #set up model
    model = JTPredict(vocab, hidden_size, latent_size, depth, stereo=stereo)
    model.create_ffn()
    
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    
    if not args.input_model_file == "":
        from jtnn_vae import JTNNVAE
        model_VAE = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)
        model_VAE.load_state_dict(torch.load(os.path.join(args.input_model_file,args.model_name)))
        model.jtnn = model_VAE.jtnn
        model.mpn = model_VAE.mpn
        model.embedding = model_VAE.embedding
        model.T_mean = model_VAE.T_mean
        model.T_var = model_VAE.T_var
        model.G_mean = model_VAE.G_mean
        model.G_var = model_VAE.G_var
        
    model.to(device)

            
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
    scheduler.step()

    if not args.filename == "":

        fname = os.path.join(args.log_path, args.filename)
        #, exist_ok=True)
        # #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        os.makedirs(fname)
        writer = SummaryWriter(fname)
        with open(os.path.join(fname, 'parameters.json'), 'w') as f:
            json.dump(vars(args), f)
        args.fname = fname
        
    
        
        
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=os.path.join(fname, args.filename + '.pth'))

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_r2, train_rmse = eval(args, model, device, train_loader, scaler, train = True)
        else:
            print("omit the training accuracy computation")
            train_r2, train_rmse = 0, 0
        val_r2, val_rmse = eval(args, model, device, val_loader, scaler)
        test_r2, test_rmse = eval(args, model, device, test_loader, scaler)

        print("train r2: %f\ntrain rmse: %f\n val r2: %f\n val rmse: %f\ntest r2: %f\ntest rmse: %f"\
              %(train_r2, train_rmse, val_r2, val_rmse, test_r2, test_rmse))

        # val_acc_list.append(val_acc)
        # test_acc_list.append(test_acc)
        # train_acc_list.append(train_acc)

        if not args.filename == "":
            writer.add_scalar('data/train r2', train_r2, epoch)
            writer.add_scalar('data/train rmse', train_rmse, epoch)

            writer.add_scalar('data/val r2', val_r2, epoch)
            writer.add_scalar('data/val rmse', val_rmse, epoch)
            writer.add_scalar('data/test r2', test_r2, epoch)
            writer.add_scalar('data/test rmse', test_rmse, epoch)

        early_stopping(val_rmse, model, epoch)

        if early_stopping.early_stop:
            print("Early stopping", epoch)
            break

        print("")
    if not args.filename == "":
        writer.close()
        with open(os.path.join(fname, 'logs.txt'), 'w') as f:
            f.write('Best epoch is '+str(epoch)+'\n')
        # torch.save(model.gnn.state_dict(), os.path.join(fname, args.filename+'.pth'))
        model.load_state_dict(torch.load(os.path.join(fname, args.filename + '.pth')))
        train_r2, train_rmse = eval(args, model, device, train_loader, scaler, train = True)
        val_r2, val_rmse = eval(args, model, device, val_loader, scaler)
        test_r2, test_rmse = eval(args, model, device, test_loader, scaler, test = True)
        with open(os.path.join(fname, 'logs.txt'), 'a') as f:
            f.write('Test RMSE is '+str(test_rmse)+'\n')
            f.write('Test R2 is '+str(test_r2)+'\n')
            f.write('Val RMSE is '+str(val_rmse)+'\n')
            f.write('Val R2 is '+str(val_r2)+'\n')
            f.write('Train RMSE is '+str(train_rmse)+'\n')
            f.write('Train R2 is '+str(train_r2)+'\n')

if __name__ == "__main__":
    main()