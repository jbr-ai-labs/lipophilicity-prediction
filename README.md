<!--
*** Thanks for checking out this README Template. If you have a suggestion that would
*** make this better, please fork the repo and create a pull request or simply open
*** an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email
-->


# Lipophilicity Prediction with Multitask Learning and Molecular Substructures Representation

Lipophilicity is one of the factors determining the permeability of the cell membrane to a drug molecule. Hence, accurate lipophilicity prediction is an essential step in the development of new drugs. We introduce a novel approach to encoding additional graph information by extracting molecular substructures. By adding a set of generalized atomic features of these substructures to an established Direct Message Passing Neural Network (D-MPNN) we were able to achieve a new state-of-the-art result at the task of prediction of two main lipophilicity coefficients, namely logP and logD descriptors. We further improve our approach by employing a multitask approach to predict logP and logD values simultaneously. Additionally, we present a study of the model performance on symmetric and asymmetric molecules, that may yield insight for further research.

The figure below shows the overall network architecture of our method named StructGNN.

![product-screenshot](imgs/WorkshopModelBW.png)

The following datasets have been used: 

| Dataset name | Number of Samples | Description | Sources |
| --- | --- | --- | --- |
| logp_wo_logp_json_wo_averaging | 13688 | All logP datasets except logp.json | Diverse1KDataset.csv, NCIDataset.csv, ochem_full.csv, physprop.csv |
| logd_Lip_wo_averaging | 4166 | Merged datasets w/o strange (very soluble) molecules and standardized SMILES. Between duplicated logD for one SMILES the most common value was chosen | Lipophilicity |
| logp_wo_logp_json_logd_Lip_wo_averaging | 17603 | Merged LogP and LogD datasets, 251 molecules have logP and logD values | logp_wo_logp_json_wo_averaging,<br/>logd_Lip_wo_averaging |

## Paper

For a detailed description of StructGNN we refer the reader to the paper ["Lipophilicity Prediction with Multitask Learning and Molecular Substructures Representation"](https://arxiv.org/abs/2011.12117).

If you wish to cite this code, please do it as follows:
```
@misc{lukashina2020lipophilicity,
      title={Lipophilicity Prediction with Multitask Learning and Molecular Substructures Representation}, 
      author={Nina Lukashina and Alisa Alenicheva and Elizaveta Vlasova and Artem Kondiukov and Aigul Khakimova and Emil Magerramov and Nikita Churikov and Aleksei Shpilman},
      year={2020},
      eprint={2011.12117},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

*Machine Learning for Molecules Workshop @ NeurIPS 2020*

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Structure of this repository](#structure-of-this-repository)
* [StructGNN](#structgnn)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Training](#training)
* [Baselines](#baselines)
  * [DMPNN](#dmpnn)
  * [OTGNN](#otgnn)
  * [JtVAE](#jtvae)


<!-- ABOUT THE PROJECT -->
## Structure of this repository

There are 3 main folders:

1. [Jupyter Notebooks with EDA, data preprocessing, predictions analysis](notebooks/)
2. [Data files](data/)
3. [Scripts for models training](scripts/SOTA)

This repository was built with the help of

* [OTGNN original repo](https://github.com/benatorc/OTGNN)
* [Junction Tree original repo](https://github.com/benatorc/OTGNN)
* [DMPNN original repo](https://github.com/chemprop/chemprop)

<!-- GETTING STARTED -->
## StructGNN

To get a local copy up and running follow these simple steps.

### Prerequisites

To use `chemprop` with GPUs, you will need:
 * cuda >= 8.0
 * cuDNN


### Installation

1. `git clone https://github.com/jbr-ai-labs/lipophilicity-prediction.git`
2. `cd scripts/SOTA/dmpnn`
3. `conda env create -f environment.yml`
4. `conda activate chemprop`
5. `pip install -e .`


<!-- USAGE EXAMPLES -->
### Training

To train the model you can either use existing DVC pipeline or run training manually.

The first step is common for both runs.
1. Set `params.yaml`

  ```
  additional_encoder: True # set StructGNN architecture
  
  file_prefix: <name of dataset without format and train/test/val prefix>
  split_file_prefix: <name of dataset without format and train/test/val prefix for `train_val_data_preparation.py` script>
  input_path: <path to split dataset>
  data_path: <path to train_val dataset>
  separate_test_path: <path to test dataset>
  save_dir: <path to experiments logs>
 
 
  epochs: <number of training epochs>
  patience: <early stopping patience>
  delta: <early stopping delta>
 
  features_generator: [rdkit_wo_fragments_and_counts]
  no_features_scaling: True
  
  target_columns: <name of target column>
 
  split_type: k-fold
  num_folds: <number of folds>
 
  substructures_hidden_size: 300
  hidden_size: 800 # dmpnn ffn hidden size
  ```
A full list of available arguments can be found in [dmpnn/chemprop/args.py](scripts/SOTA/dmpnn/chemprop/args.py)

#### Manual run

2. Run `python ./scripts/SOTA/dmpnn/train_val_data_preparation.py` - to create dataset for cross-validation procedure
3. Run `python ./scripts/SOTA/dmpnn/train.py --dataset_type regression --config_path_yaml ./params.yaml` - to train model

#### DVC run

2. Run `dvc repro` command

## Baselines

### DMPNN

Article - [Analyzing Learned Molecular Representations for Property Prediction](https://arxiv.org/pdf/1904.01561v5.pdf)

Original Github Repo - https://github.com/chemprop/chemprop

#### Requirements

All the requirements are the same as for StructGNN

#### Training

The training procedure is the same as [StructGNN](#StructGNN), but set `additional_encoder: False` in `params.yaml`

### OTGNN

Article - [Optimal Transport Graph Neural Networks](https://arxiv.org/pdf/2006.04804v2.pdf)

Original Github Repo - https://github.com/benatorc/OTGNN

#### Requirements

``` 
conda create -n mol_ot python=3.6.8
sudo apt-get install libxrender1

conda install pytorch torchvision -c pytorch
conda install -c rdkit rdkit
conda install -c conda-forge pot
conda install -c anaconda scikit-learn
conda install -c conda-forge matplotlib
conda install -c conda-forge tqdm
conda install -c conda-forge tensorboardx
```

#### Data

Prepara data and splits with [1_data_preparation.ipynb notebook](notebooks/models_postprocessing/otgnn/1_data_preparation.ipynb)

#### Training

Running cross-validation:

```cd ./scripts/SOTA/otgnn/; python train_proto.py -data logp_wo_json -output_dir output/exp_200 -lr 5e-4 -n_splits 5 -n_epochs 100 -n_hidden 50 -n_ffn_hidden 100 -batch_size 16 -n_pc 20 -pc_size 10 -pc_hidden 5 -distance_metric wasserstein -separate_lr -lr_pc 5e-3 -opt_method emd -mult_num_atoms -nce_coef 0.01```

### JtVAE

Article - [Junction Tree Variational Autoencoder for Molecular Graph Generation](https://arxiv.org/pdf/1802.04364%5D)

Original Github Repo - https://github.com/wengong-jin/icml18-jtnn

#### Requirements

``` 
conda create -n jtree python=2.7

conda install pytorch torchvision -c pytorch
conda install -c rdkit rdkit
conda install -c anaconda scikit-learn
conda install -c conda-forge matplotlib
conda install -c conda-forge tqdm
conda install -c conda-forge tensorboardx
```


Original article proposed to train autoencoder architecture for reconstruction task, here we use only encoder part in regression task.

#### Data

To run with prepared data:

Download pickle-file [SMILES_TO_MOLTREE.pickle](https://drive.google.com/file/d/15e8Tq0xKwIVUpizKmn-o3xznq0iBunvp/view?usp=sharing) from gdrive and place it to `data/raw/baselines/jtree/` directory.

NB!

JTree Vocabulary can lead to exceptions in case of unknown substructures. To skip such molecules run [2_encode_molecules.ipynb](notebooks/models_postprocessing/jtree/2_encode_molecules.ipynb) with appropriate data. It will save nesessary files in ```data/raw/baselines/jtree/train_errs.txt(val_errs.txt, test_errs.txt)```. 
 

#### Training

Running with best parameters:

```cd ./scripts/SOTA/jtree/; python train_encoder_more_atom_feats_CV.py --filename "exp" --epochs 200 --patience 35 --vocab_path '../../../data/raw/baselines/jtree/vocab.txt' --file_prefix logp_wo_logp_json_wo_averaging```

### Morgan Fingerprints

[Jupyter Notebooks with model and analysis of predictions](notebooks/models_postprocessing/count_morgan_fingerprint)
