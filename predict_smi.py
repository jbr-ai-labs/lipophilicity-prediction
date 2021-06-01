"""Preprocess .smi file and run logP and logD predictions."""

import pathmagic
from make_predictions import make_predictions
from multiprocessing.pool import Pool
from standardize_smiles import StandardizeTautomers
from tqdm import tqdm
from args import PredictArgs
import json





class SmiParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.molecules = []
        self.smiles = []
        self.names = []

    def read_molecules(self):
        with open(self.filepath, 'r') as f:
            data = f.readlines()
            if len(data[-1]) == 0:
                data = data[:-1]
            self.smiles = list(map(lambda x: x.split()[0], data))
            self.names = list(map(lambda x: x.split()[1], data))
        self.smiles = self.standardize_smiles(self.smiles)
        self.molecules = self.create_smi_dict()
        return self.molecules

    def create_smi_dict(self):
        self.molecules = [{'smiles': smiles, 'name': name} for smiles, name in zip(self.smiles, self.names)]
        return self.molecules

    @staticmethod
    def standardize_smiles(smiles):
        tautomer_stand = StandardizeTautomers()
        with Pool(10) as pool:
            res = list(
                tqdm(pool.imap(tautomer_stand.standardize_smiles, smiles), total=len(smiles))
            )
        return res


if __name__ == '__main__':
    smi_parser = SmiParser('test.smi')
    molecules = smi_parser.read_molecules()
    predictions = []
    for mol in molecules:
        smiles = mol['smiles']
        preds = make_predictions(args=PredictArgs().parse_args(), smiles=[smiles])
        mol['logp'] = preds[0][0]
        mol['logd'] = preds[0][1]

        predictions.append(mol)
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)
