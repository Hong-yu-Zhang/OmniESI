# CatPred-DB tasks
## Ensemble Test: Align with CatPred (https://www.nature.com/articles/s41467-025-57215-9)
echo "Training CatPred-DB kcat task"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 --standalone main_ensemble.py --model configs/model/OmniESI_ensemble.yaml --data configs/data/CatPred_kcat.yaml --task regression

echo "Training CatPred-DB km task"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 --standalone main_ensemble.py --model configs/model/OmniESI_ensemble.yaml --data configs/data/CatPred_km.yaml --task regression

echo "Training CatPred-DB ki task"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 --standalone main_ensemble.py --model configs/model/OmniESI_ensemble.yaml --data configs/data/CatPred_ki.yaml --task regression



# ESP task
echo "Training ESP task"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 --standalone main.py --model configs/model/OmniESI.yaml --data configs/data/esp.yaml --task binary



# Mutational effect tasks
echo "Training epistasis_amp task"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 --standalone main.py --model configs/model/OmniESI.yaml --data configs/data/epistasis_amp.yaml --task binary

echo "Training epistasis_amp task"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 --standalone main.py --model configs/model/OmniESI.yaml --data configs/data/epistasis_ctx.yaml --task binary

echo "Training epistasis_amp task"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 --standalone main.py --model configs/model/OmniESI.yaml --data configs/data/mut_classify.yaml --task binary



# Ablation study on CatPred-DB_kact and kat_km tasks
echo "Ablation study on CatPred-DB_kact"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 --standalone main_ensemble.py --model configs/model/Baseline_ensemble.yaml --data configs/data/CatPred_kcat.yaml --task regression
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 --standalone main_ensemble.py --model configs/model/BCFM_ensemble.yaml --data configs/data/CatPred_kcat.yaml --task regression
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 --standalone main_ensemble.py --model configs/model/CCFM_ensemble.yaml --data configs/data/CatPred_kcat.yaml --task regression

echo "Ablation study on kat_km"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 --standalone main.py --model configs/model/Baseline.yaml --data configs/data/kcat_km.yaml --task regression
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 --standalone main.py --model configs/model/BCFM.yaml --data configs/data/kcat_km.yaml --task regression
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 --standalone main.py --model configs/model/CCFM.yaml --data configs/data/kcat_km.yaml --task regression
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 --standalone main.py --model configs/model/OmniESI.yaml --data configs/data/kcat_km.yaml --task regression