import json
from collections import defaultdict
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Dict, List, Optional

import os
from loguru import logger
from rdkit import Chem
from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer
from tqdm import tqdm
import pandas as pd


class DatasetsHolder:
    @staticmethod
    def read_datasets(inp_folder_path):
        with os.scandir(inp_folder_path) as entries:
            return dict([(entry.name, pd.read_csv(entry, index_col=0)) for entry in entries if entry.is_file()])


class StandardizeDatasets:
    @staticmethod
    def standardize_smiles(smi: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        else:
            logger.info("Smiles {} was not 1_filtering. Return None", smi)
            return None

    @logger.catch()
    def standardize(self, inp_path: Path, out_path: Path):
        datasets = DatasetsHolder.read_datasets(inp_path)
        failed_smiles: Dict[str, List[str]] = defaultdict(list)

        for name, dataset in datasets.items():
            logger.info("Processing {}", name)
            with Pool(10) as pool:
                res = list(
                    tqdm(pool.imap(self.standardize_smiles, dataset.smiles), total=dataset.shape[0])
                )

            failed_smiles[name].append(dataset.smiles[[smi is None for smi in res]].tolist())

            dataset["smiles"] = res
            dataset = dataset[~dataset.smiles.isna()]
            dataset.to_csv(out_path / f"{name}", index=False)

        with (out_path / "failed_smiles.json").open("w") as f:
            json.dump(failed_smiles, f, indent=2)


class StandardizeTautomers(StandardizeDatasets):
    @staticmethod
    def standardize_smiles(smi: str) -> Optional[str]:
        tc = TautomerCanonicalizer()
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(tc.canonicalize(mol))
        else:
            logger.info("Smiles {} was not 1_filtering. Return None", smi)
            return None