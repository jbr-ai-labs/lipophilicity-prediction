FROM conda/miniconda3

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get install -y libxrender1

RUN mkdir -p /lipophilicity-prediction

WORKDIR /lipophilicity-prediction

COPY . /lipophilicity-prediction

RUN conda update -n base -c defaults conda

RUN conda env update --name base --file environment.yml

RUN pip install git+https://github.com/bp-kelley/descriptastorus
