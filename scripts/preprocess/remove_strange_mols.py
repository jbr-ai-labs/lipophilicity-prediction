from pathlib import Path

import click
import pandas as pd
from loguru import logger
from rdkit import Chem


class DropExtraMol:
    def __init__(self):
        self.strange_molecules = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CO")]
        self.allowed_atoms = {
            "C",
            "H",
            "I",
            "Br",
            "F",
            "O",
            "Mg",
            "Cl",
            "N",
            "P",
            "S",
            "B",
            "Na",
            "K",
            "Ca",
            "Fe",
        }

    def check_not_strange_mols(self, mol_smi: str) -> bool:
        """
        Неограниченно растворимые в воде соединения занесены в :py:attr:`strange_molecules`,
        если молекула из этого списка, то функция возвращает ``False``
        """
        mol = Chem.MolFromSmiles(mol_smi)
        for strange_mol in self.strange_molecules:
            if len(mol.GetAtoms()) == len(strange_mol.GetAtoms()):
                if mol.HasSubstructMatch(strange_mol) and strange_mol.HasSubstructMatch(mol):
                    logger.debug(
                        "The molecule {} is in the list of very soluble in water molecules", mol_smi
                    )
                    return False
        logger.debug(
            "The molecule {} isn't in the list of very soluble in water molecules", mol_smi
        )
        return True

    def check_allowed_atoms(self, mol_smi: str) -> bool:
        """
        Проверяет наличие допустимых атомов в молекуле. Если в молекуле есть атомы,
        вне этого списка, то возвращает ``False``
        """
        mol = Chem.MolFromSmiles(mol_smi)
        if any(atom.GetSymbol() not in self.allowed_atoms for atom in mol.GetAtoms()):
            logger.debug(
                "The molecule {} has the atom that out of the list of allowed atoms", mol_smi
            )
            return False
        logger.debug("All atoms in the molecule {} are allowed", mol_smi)
        return True

    @staticmethod
    def check_atoms(mol_smi: str, threshold: int = 5) -> bool:
        """
        Подсчитывает количество атомов в молекуле. Если атомов меньше, чем ``threshold``,
        то возвращает ``False``
        """
        mol = Chem.MolFromSmiles(mol_smi)
        if len(mol.GetAtoms()) < threshold:
            logger.debug("The molecule {} has less than {} atoms", mol_smi, threshold)
            return False
        logger.debug("The molecule {} has more than {} atoms", mol_smi, threshold)
        return True


def remove_strange_mols(input_path: Path, output_path: Path, threshold: int = 5) -> None:
    df = pd.read_csv(input_path)
    drop_extra_mols = DropExtraMol()

    df["not_strange_mols"] = df["smiles"].apply(drop_extra_mols.check_not_strange_mols)
    df["allowed_atoms"] = df["smiles"].apply(drop_extra_mols.check_allowed_atoms)
    df["counted_atoms"] = df["smiles"].apply(drop_extra_mols.check_atoms)

    df_without_strange = df[df["not_strange_mols"]]
    logger.info("Shape of the dataset without strange molecules: {}", df_without_strange.shape)
    df_with_allowed_atoms = df_without_strange[df_without_strange["allowed_atoms"]]
    logger.info("Shape of the dataset with allowed atoms: {}", df_with_allowed_atoms.shape)
    df_counted_atoms = df_with_allowed_atoms[df_with_allowed_atoms["counted_atoms"]]
    logger.info(
        "Shape of the dataset with molecules that have more than {} atoms: {}",
        threshold,
        df_counted_atoms.shape,
    )
    df_counted_atoms.drop(["not_strange_mols", "allowed_atoms", "counted_atoms"], axis=1).to_csv(
        output_path, index=False
    )


@logger.catch()
@click.command()
@click.argument("in_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path(exists=False))
def main(in_path, out_path):
    in_path_ = Path(in_path)
    out_path_ = Path(out_path)
    remove_strange_mols(in_path_, out_path_)


if __name__ == "__main__":
    main()