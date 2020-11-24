## Table of Contents

### Data EDA and Preprocessing

1. [EDA_of logP logp.json dataset](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/1_eda_logp_json.ipynb)
2. [EDA of logP physprop dataset](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/2_eda_physprop.ipynb)
3. [General usage of standartization scripts](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/3_standardize.ipynb)
4. [EDA of logP OChem dataset](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/4_eda_ochem_dataset.ipynb)
5. [EDA of logP Diverse dataset](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/5_eda_DiverseDataset_dataset.ipynb)
6. [EDA OF logP NCI dataset](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/6_eda_NCI_dataset.ipynb)
7. [Standartization of SMILES in datasets and merging all data sources of logP](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/7_standardize_merge.ipynb)
    * [Standartization of SMILES in datasets and merging all data sources of logP except logP.json](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/7.5_standardize_merge_wo_logp_json.ipynb)
8. [EDA of merged logP with pH and Temperature dataset](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/8_eda_logP_ph_T_datasets.ipynb)
9. [EDA of logD benchmark Lipophilicity dataset](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/9_eda_logD_Lipophilicity_dataset.ipynb)
10. [EDA of logD logD7.4 dataset](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/10_eda_logD_logd74_dataset.ipynb)
11. [EDA of logD OChem dataset](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/11_eda_logD_ochem_dataset.ipynb)
12. [Standartization of SMILES in datasets and merging all data sources of logD](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/12_merge_standardize_logD_datasets.ipynb)
13. [Analysis of logP+pH dataset](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/13_full_logP_pH_dataset_analysis.ipynb)
14. [Analysis of logP dataset with averaging of duplicated values (pH and Temperature were dropped)](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/14_full_logP_dataset_analysis.ipynb)
15. [Analysis of logP dataset with averaging of duplicated values (pH and Temperature were not known)](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/15_logP_without_parameters_dataset.ipynb)
16. [Split logp_mean, logP_pH_range_mean, logP_wo_parameters and logD_pH datasets to train/val/test](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/16_datasets_split.ipynb)
17. [Split logP dataset without averaging of duplicated values and logD (only Lipophilicity source) datasets](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/17_create_datasets_wo_averaging_split.ipynb)
18. [Algorithm for defining symmetric and asymetric molecules](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/18_symmetry_analysis.ipynb)
19. [Split dataset with ZINC molecules](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/19_zinc_dataset_splitting.ipynb)
20. [Selecting molecules with specific properties to test model](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/20_selecting_molecules_for_logP_measurement.ipynb)
21. [Calculate the percent of atoms in more than one ring in logP dataset](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/21_atoms_in_rings_analysis.ipynb)
22. [Creating of MultiTask datasets and their splits](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/22_MultiTask_Datasets_merge_split.ipynb)
23. [EDA of benchmark ESOL anf FreeSolv datasets](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/23_FreeSolv_ESOL_eda.ipynb)
24. [Standartization, merge and split of benchmark ESOL and FreeSolv datasets](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/24_FreeSolv_ESOL_standartization_remove_strange_mols_no_averaging_split.ipynb)
25. [Split of final logp_wo_logp_json_wo_averaging and logd_Lip_wo_averaging datasets](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/25_new_dataset_splits.ipynb)

### Analysis of molecules with hyper-atoms

1. [Count the percent of molecules which were merged strongly (length of molecule significantly decreased)](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/no_ring_molecule_analysis/1_check_custom_molecule_size.ipynb)
2. [Count the unique representations of molecules and get the most common ones](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/no_ring_molecule_analysis/2_check_unique_hashes.ipynb)

### Models notebooks

1. [Analyzing the best and the worst predictions](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/models_postprocessing/analyzing_models.ipynb)
2. [Testing substructures extraction](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/models_postprocessing/dmpnn_substructures_check.ipynb)
3. [Comparison of StructGNN and D-MPNN predictions](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/models_postprocessing/3_DMPNN_StructGNN_models_analysis.ipynb)

#### StructGNN and D-MPNN

1. [Merging train and validation datasets for cross-validation](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/models_postprocessing/dmpnn/1_data_preparation.ipynb)
2. [Analyzing the best and the worst predictions](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/models_postprocessing/dmpnn/2_get_worst_and_best_predictions.ipynb)
3. [Analysis of additional RDKit features in model](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/models_postprocessing/dmpnn/3_RDKit_features_analyzis.ipynb)
4. [RDKit features + XGBoost and RDKit features + MLPRegressor models](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/models_postprocessing/dmpnn/4_RDKit_features_regression.ipynb)

#### Count Morgan Fingerprint

1. [Morgan Fingerprint + FFNN model and hyperparameter optimization](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/models_postprocessing/count_morgan_fingerprint/1_morgan_fingerprints_neural_network_hyperparameters_tuning.ipynb)
2. [Analyzing the best and the worst predictions](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/models_postprocessing/count_morgan_fingerprint/2_analyzing_morgan_models.ipynb)
3. [Morgan Fingerprint + FFNN model with cross-validation](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/models_postprocessing/count_morgan_fingerprint/3_cross_validatioin.ipynb)

#### OTGNN

1. [Prepare data for training](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/models_postprocessing/otgnn/1_data_preparation.ipynb)
2. [Analyzing the best and the worst predictions](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/models_postprocessing/otgnn/2_get_worst_and_best_predictions.ipynb)

#### JtVAE

1. [Get list of all SMILES in data](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/models_postprocessing/jtree/1_convert_list_of_SMILES.ipynb)
2. [Get SMILES of molecules containing substructures that are not presented in JTree vocabulary](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/models_postprocessing/jtree/2_encode_molecules.ipynb)
3. [DRAFT Notebook with training of JtVAE encoder part + FFNN](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/models_postprocessing/jtree/3_train_encoder_predictor.ipynb)
4. [Feature importance of fingerprint extracted by pretrained JtVAE encoder](https://github.com/jbr-ai-labs/lipophilicity-prediction/blob/SOTA/notebooks/models_postprocessing/jtree/4_ridge_regression_feature_importance.ipynb)

