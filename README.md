# MESI
The official code repository of "Multi-purpose enzyme-substrate interaction prediction with progressive conditional deep learning"

# News
⭐**Nov 27, 2024:** The model weights for **all datasets and splits** are released!

⭐**Nov 26, 2024:** The source code for MESI is released!

# Overview
Understanding and modeling enzyme-substrate interactions is crucial for designing enzymes with tailored functions, thereby advancing the field of enzyme engineering. The diversity of downstream tasks related to enzyme catalysis calls for a computational architecture that actively perceives enzyme-substrate interaction patterns to make unified predictions for multiple objectives. Here, we introduce MESI, a progressive conditional deep learning framework for multi-purpose enzyme-substrate interaction prediction. By decomposing the modeling of enzyme-substrate interactions into a two-stage process, MESI incorporates two conditional networks that respectively emphasize enzymatic reaction specificity and crucial catalytic interactions, facilitating a gradual shift in the feature latent space from the general domain to the catalysis-aware domain. Across various downstream tasks, MESI consistently outperforms state-of-the-art methods on top of a unified architecture. Furthermore, the proposed conditional networks implicitly capture the fundamental patterns of enzyme catalysis with negligible additional computational overhead, as evidenced by extensive ablation experiments. With the support of this conditional perception mechanism, MESI enables cost-effective and accurate identification of active sites without requiring any structural information, highlighting enzyme residues and substrate functional groups involved in diverse and critical catalytic interactions. Overall, MESI represents a unified prediction paradigm for downstream tasks related to enzyme catalysis, paving the way for deep-learning-based catalytic mechanism cracking and enzyme engineering with strong generalization and interpretability.
Source code for MESI is released!
![overview](./figure/Fig1_overview.png)
# Installation
Create a new environment for MESI:
```shell
conda create -n MESI python=3.8
conda activate MESI
```
Installation for pytorch 1.12.1:
```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
Installation for other dependencies:
```shell
pip install -r requirements.txt
```

# Repository content
```shell
- configs/ # configuration folder
    - data/ # dataset configuration
    - model/ # model configuration

- data_process/ # scripts for generating ESM embeddings
    - embedding_1fold.py # for single(fixed)-fold dataset
    - embedding_5fold.py # for five-fold dataset
    - partion_data.py # generate five-fold dataset with embeddings

- datasets/ # ESI datasets
    - single-fold dataset/
        - train.csv
        - val.csv
        - test.csv
    - five-fold dataset/
        - [DATASET_NAME].csv

- module/ # source code for the modules in MESI
    - CN.py # conditional networks (BCFM & CCFM)
    - Encoder.py # encoders for substrate and enzyme
    - Transformer.py # transformer blocks

- results/ # storation of model weights and training results
    - [DATASET_NAME]/
        - MESI/ # released weights for MESI 
        - exp0/
        ...
        - expn/ # path to the training results once start a new training

- run/ # trainer & DDP trainer

- visualization/ # source code for visualization
    - PCA_analysis/ # PCA for mut_classify and kcat/Km 
    - Interpretability_analysis/ # interpretability for pocket and active site discovery
        - AF3_complex/ # enzyme-substrate complex predicted by AF3

- config.py # basic configuration

- utils.py # function utils

- dataloader.py

- main.py # script for training

- models.py # main architecture of MESI

- test.py # script for testing
```

# Data preparation

## Download data
Download ESI datasets and model weights at: https://zenodo.org/records/14224548


## Prepare ESI datasets
Make sure all the ESI datasets are stored under:
```shell
/.../MESI/datasets/*
```

## Prepare model weights of MESI
We have released all the weights from the single-fold and five-fold experiments for the main ESI tasks. Make sure all the weights are stored under:
```shell
/.../MESI/results/*
```

## Obtain ESM-2(650M) embeddings
Generate embeddings for all ESI datasets. First specify the storage path for the ESM embeddings: [DATA_PATH]. The following script will save the embeddings to `[DATA_PATH]/dataset_name/esm/`. By default, this will retrieve the ESM features for all datasets in the MESI paper. We recommend running it on a GPU with at least 24GB memory
```python
cd ./data_process
python embedding_1fold.py --feat_dir [DATA_PATH]
python embedding_5fold.py --feat_dir [DATA_PATH]
```
!!NOTE: We highly recommand employ the absolute path `/.../MESI/datasets_embeddings` for [DATA_PATH]. This will ensure the quick usage of subsequent code and the reproducibility of results.

## Obtain the partitioned datasets
Run `partion_data.py` to obtain the five-fold datasets:
```python
python partion_data.py --dataset_dir [DATA_PATH]
```
The final dataset is organized according to the following example:
```shell
    [DATA_PATH]/DLKcat/esm #for five-fold dataset
    [DATA_PATH]/DLKcat_1/train,val,test.csv
    [DATA_PATH]/DLKcat_2/train,val,test.csv
    ...
    [DATA_PATH]/DLKcat_5/train,val,test.csv

    ......

    [DATA_PATH]/esp/esm,train,val,test.csv #for single-fold dataset

    ......
```


# Quick reproduction

Once the ESM-2 embeddings and all model weights are prepared, you can quickly reproduce the main experimental results mentioned in the paper using the `reproduce.sh` script:
```shell
cd /.../MESI
chmod +x reproduce.sh
./reproduce.sh
```
On a single V100 GPU, it takes approximately one hour to obtain all the results.



# MESI for regression tasks

## Execution templates
For regression tasks such as kcat, Km, and kcat/Km, the basic execution templates for testing and training are as follows:
```python
#test
python test.py \
    --model configs/model/[MODEL].yaml \
    --data configs/data/[DATA].yaml \
    --weight ./results/[DATA]/[MODEL]/best_model_epoch.pth \
    --task regression

#train (single GPU)
python main.py \
    --model configs/model/[MODEL].yaml \
    --data configs/data/[DATA].yaml \
    --task regression

#train (multiple GPUs)
CUDA_VISIBLE_DEVICES=[0,1,2,3] python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --model configs/model/[MODEL].yaml \
    --data configs/data/[DATA].yaml \
    --task regression #For example, 4 GPUs 
```
## Configuration choice
Below are the configuration options involved in the regression tasks:
```python
[MODEL]: MESI, CCFM, BCFM, Baseline

# For kcat prediction
[DATA]: DLKcat_1, DLKcat_2, DLKcat_3, DLKcat_4, DLKcat_5 #5 fold experiments for DLKcat
[DATA]: MPEK_kcat_1, MPEK_kcat_2, MPEK_kcat_3, MPEK_kcat_4, MPEK_kcat_5 #5 fold experiments for MPEK_kcat
[DATA]: CatPred_kcat

# For Km prediction
[DATA]: GraphKM_1, GraphKM_2, GraphKM_3, GraphKM_4, GraphKM_5 #5 fold experiments for GraphKM
[DATA]: MPEK_km_1, MPEK_km_2, MPEK_km_3, MPEK_km_4, MPEK_km_5 #5 fold experiments for MPEK_km
[DATA]: CatPred_km

# For kcat/Km prediction
[DATA]: MPEK_kcat_km
```


# MESI for binary classification tasks
## Execution templates
For binary classification tasks, such as enzyme-substrate pair prediction, epistasis prediction, and single mutation classification, the basic execution templates for training and testing are as follows:
```python
#test
python test.py \
    --model configs/model/[MODEL].yaml \
    --data configs/data/[DATA].yaml \
    --weight ./results/[DATA]/[MODEL]/best_model_epoch.pth \
    --task binary

#train (single GPU)
python main.py \
    --model configs/model/[MODEL].yaml \
    --data configs/data/[DATA].yaml \
    --task binary

#train (multiple GPUs)
CUDA_VISIBLE_DEVICES=[0,1,2,3] python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --model configs/model/[MODEL].yaml \
    --data configs/data/[DATA].yaml \
    --task binary #For example, 4 GPUs 
```

## Configuration choice
Below are the configuration options involved in the binary classification tasks:
```python
[MODEL]: MESI

# For enzyme-substrate pair prediction
[DATA]: esp 

# For epistasis prediction
[DATA]: epistasis_amp
[DATA]: epistasis_ctx

# For single mutation classification
[DATA]: mut_classify
```

# Visualization results reproduction

## PCA analysis
We have stored the embeddings used for PCA visualization of the kcat/Km and single mutation classification tasks in `/.../MESI/visualization/PCA_analysis`. You can easily reproduce the results from our paper by running the following command:
```python
cd /.../MESI/visualization/PCA_analysis
python plot_kcat_km.py
python plot_mut.py
```
The PCA visualization images will be stored in the `fig_pca` folder.

## Interpretability analysis
We demonstrate MESI's sensitivity to the residues of enzyme and functional groups of substrate that are critical for catalysis, through statistical analysis and specific case study.

### Statistical analysis

Run `att_statistic.py` to obtain the mean attention score of the residues in active site and other region of the enzyme in each enzyme-substrate complex:
```shell
cd /.../MESI/visualization/Interpretability_analysis
python att_statistic.py
```

Run `att_ratio.py` to obtain the distribution of the ratio between the mean attention scores of active site region and other region residues, providing a clearer visualization of MESI's increased focus on the residues in active site region:
```shell
python att_ratio.py
```

### Specific case study
You can also use `PyMOL` to visualize the residue attention score of enzyme in a specific case:
```python
python show_enzyme_att.py \
    --split [1-5] \
    --index [index_num] \
    --model [Baseline, BCFM, CCFM, MESI] \
    --show_surface \ # (Optional) We recommend not using this argument to prevent occlusion.
```
The script utilized to visualize the atom attention score of substrate is also available:
```python
python show_substrate_att.py \
    --split [1-5] \
    --index [index_num] \
```
