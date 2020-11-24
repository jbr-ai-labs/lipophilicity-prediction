import json
import os
import pandas as pd
from rdkit import Chem
from pathlib import Path

from standardize_smiles import StandardizeTautomers
from remove_strange_mols import remove_strange_mols

if __name__ == '__main__':
    """
    Runs big datasets preprocessing (standardization and removing strange molecules).
    """
    INPUT_PATH = Path("../../data/1_filtering/smiles_only")
    OUTPUT_PATH = Path("../../data/2_standardize/smiles_only")

    StandardizeTautomers().standardize(INPUT_PATH, OUTPUT_PATH)
    atom_counts_mean = {}
    with os.scandir(OUTPUT_PATH) as entries:
        for entry in entries:
            remove_strange_mols(entry, entry)
            data = pd.read_csv(entry)
            atom_counts = pd.Series([Chem.MolFromSmiles(s).GetNumAtoms() for s in data.smiles])
            atom_counts_mean[entry] = atom_counts.mean()
    with (OUTPUT_PATH / "datasets_mean_len.json").open("w") as f:
        json.dump(atom_counts_mean, f, indent=2)
