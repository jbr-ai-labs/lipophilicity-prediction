FROM conda/miniconda3

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get install -y libxrender1

RUN mkdir -p /lipophilicity-prediction

WORKDIR /lipophilicity-prediction

COPY . /lipophilicity-prediction

RUN conda update -n base -c defaults conda

RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "lipophilicity-prediction36", "/bin/bash", "-c"]

RUN pip install git+https://github.com/bp-kelley/descriptastorus

RUN python predict_smi.py --test_path ./test.smi --checkpoint_path ./model.pt --features_generator rdkit_wo_fragments_and_counts --additional_encoder --no_features_scaling