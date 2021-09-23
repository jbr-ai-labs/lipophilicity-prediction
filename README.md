# lipophilicity-prediction

This project was created in collaboration with HSE and forked from [this repo](https://github.com/jbr-ai-labs/lipophilicity-prediction).

## Run local build

### Build an Image from the `Dockerfile`
```
docker build -t lipophilicity-prediction .
```

### Run a Docker Container from the Image

The lipophilicity prediction pretrained model is placed at `model.pt` file.

To evaluate the model, run a Docker Container by following command:
```
docker run lipophilicity-prediction bash -c '$(conda run python -c "import sys;print(sys.executable)") predict_smi.py --test_path ./test.smi --checkpoint_path ./model.pt --features_generator rdkit_wo_fragments_and_counts --additional_encoder --no_features_scaling'
```

where `test.smi` is a file with molecules for predictions. Each row of `test.smi` is SMILES of molecules and
molecule's name, separated with a space character.

Script produces file `predictions.json` with format `List[ResDict]`.

ResDict:

```
smiles: str
name: str
logp: float
logd: float
```
