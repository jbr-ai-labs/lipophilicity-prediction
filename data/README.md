# Datasets

Final datasets are stored [here](3_final_data/).

| Dataset name | Number of Samples | Description | Sources |
| --- | --- | --- | --- |
| logp_wo_logp_json_wo_averaging | 13688 | All logP datasets except logp.json | Diverse1KDataset.csv, NCIDataset.csv, ochem_full.csv, physprop.csv |
| logd_Lip_wo_averaging | 4166 | Merged datasets w/o strange (very soluble) molecules and standardized SMILES. Between duplicated logD for one SMILES the most common value was chosen | Lipophilicity |
| logp_wo_logp_json_logd_Lip_wo_averaging | 17603 | Merged LogP and LogD datasets, 251 molecules have logP and logD values | logp_wo_logp_json_wo_averaging,<br/>logd_Lip_wo_averaging |
