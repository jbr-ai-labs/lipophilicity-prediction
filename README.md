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

![product-screenshot](https://raw.githubusercontent.com/jbr-ai-labs/lipophilicity-prediction/main/imgs/WorkshopModelBW.png)

The following datasets have been used: 

| Dataset name | Number of Samples | Description | Sources |
| --- | --- | --- | --- |
| logp_wo_logp_json_wo_averaging | 13688 | All logP datasets except logp.json | Diverse1KDataset.csv, NCIDataset.csv, ochem_full.csv, physprop.csv |
| logd_Lip_wo_averaging | 4166 | Merged datasets w/o strange (very soluble) molecules and standardized SMILES. Between duplicated logD for one SMILES the most common value was chosen | Lipophilicity |
| logp_wo_logp_json_logd_Lip_wo_averaging | 17603 | Merged LogP and LogD datasets, 251 molecules have logP and logD values | logp_wo_logp_json_wo_averaging,<br/>logd_Lip_wo_averaging |

- Units
    * LogP: unitless
    * LogD: unitless
- Description:
    * LogP: <img src="https://render.githubusercontent.com/render/math?math=log_{10}\frac{Concentration_{in\ octanol}^{un-ionized}}{Concentration_{in\ water}^{un-ionized}}">
    * LogD: <img src="https://render.githubusercontent.com/render/math?math=log_{10}\frac{\sum_{ionized\ forms}Concentration_{in\ octanol}^{ionized}}{\sum_{ionized\ forms}Concentration_{in\ water}^{ionized}}">


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


This repository was built with the help of
* [DMPNN original repo](https://github.com/chemprop/chemprop)

<!-- GETTING STARTED -->
## StructGNN

To get a local copy up and running follow these simple steps.


### Installation

1. `git clone https://github.com/jbr-ai-labs/lipophilicity-prediction.git`
2. `git checkout docker_evaluation`
3. `docker build -t lipophilicity-prediction .`



<!-- USAGE EXAMPLES -->
### Evaluation

To evaluate the pretrained model `model.pt` run the following command:
```
docker run lipophilicity-prediction /usr/local/envs/lipophilicity-prediction36/bin/python predict_smi.py --test_path ./test.smi --checkpoint_path ./model.pt --features_generator rdkit_wo_fragments_and_counts --additional_encoder --no_features_scaling
```

where `test.smi` - file with molecules for predictions.
Each row of `test.smi` is smiles and molecule's name, separated with ' '.

Script produces file `predictions.json` with format `List[ResDict]`.

ResDict:
```
smiles: str
name: str
logp: float
logd: float
```

