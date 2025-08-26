# OmniESI

<a href='https://arxiv.org/abs/2506.17963'><img src='https://img.shields.io/badge/ArXiv-2412.18597-red'></a> 

The official code repository of "OmniESI: A unified framework for enzyme-substrate interaction prediction with progressive conditional deep learning"

# News
⭐**Aug 26, 2025:** We have released the code, model weights and dataset for active site prediction! Check **OmniESI_site.zip** for more details.😊

⭐**Jul 10, 2025:** The source code for OmniESI is released!

⭐**Jun 25, 2025:** More downstream applications coming in one week. Stay tuned!😊


# Overview
![overview](./figure/Fig1.png)

# Installation
Create a new environment for OmniESI:
```shell
conda create -n OmniESI python=3.8 -y
conda activate OmniESI
```
Installation for pytorch 1.12.1:
```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
Installation for other dependencies:
```shell
pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html
chmod +x scripts/setup_env.sh
bash ./scripts/setup_env.sh
```

# Data preparation
Download ESI datasets and model weights at:

[Download link](https://drive.google.com/file/d/1qKmu476De75LD1EsmCU-s8SM4Cj9LoQ4/view?usp=drive_link)


## Prepare ESI datasets
Make sure all the ESI datasets are stored under:
```shell
/.../OmniESI/datasets/*
```

## Prepare model weights of OmniESI
We have released all the weights for the main ESI tasks. Make sure all the weights are stored under:
```shell
/.../OmniESI/results/*
```

## Obtain ESM-2(650M) embeddings
Generate embeddings for all ESI datasets. First specify the storage path for the ESM embeddings: [DATA_PATH]. The following script will save the embeddings to `[DATA_PATH]/dataset_name/esm/`. By default, this will retrieve the ESM features for all datasets in the OmniESI paper. We recommend running it on a GPU with at least 24GB memory
```python
cd ./scripts
python embedding.py --feat_dir [DATA_PATH]
cd ..
```
!!NOTE: We highly recommand employ the absolute path `/.../OmniESI/datasets_embeddings` for [DATA_PATH]. This will ensure the quick usage of subsequent code and the reproducibility of results.


# Quick reproduction
Once the ESM-2 embeddings and all model weights are prepared, you can quickly reproduce the main experimental results mentioned in the paper

## Inference reproduction
```shell
chmod +x ./scripts/reproduce_inference.sh
./scripts/reproduce_inference.sh
```
On a single V100 GPU, it takes approximately one hour to obtain all the results.

## Training reproduction
```shell
chmod +x ./scripts/reproduce_training.sh
./scripts/reproduce_training.sh
```
On 4 V100 GPU, it takes approximately 2-3 days to obtain all the results.


# Custom data inference
You can specify any model weights to perform inference on custom data.
First, you need to organize the enzyme-substrate pairs (enzyme sequences and substrate SMILES) you want to predict into a CSV file containing two columns: "SMILES" and "Protein".
```shell
Protein,SMILES,
seq1,smile1,
seq2,smile2,
...
```
Then, select the model weights corresponding to the specific task for inference. For example, use the weights from the ESP dataset for pair prediction.
```python
CUDA_VISIBLE_DEVICES=0 python inference.py \
                 --model configs/model/OmniESI.yaml \
                 --csv_data [INPUT CSV] \
                 --weight ./results/esp/OmniESI/best_model_epoch.pth \
                 --task binary
```
The prediction csv results will be saved in the `OmniESI/` directory.
