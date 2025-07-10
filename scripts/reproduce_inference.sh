# CatPred-DB tasks
## Ensemble Test: Align with CatPred (https://www.nature.com/articles/s41467-025-57215-9)
echo "Inference CatPred-DB kcat task"
CUDA_VISIBLE_DEVICES=0 python test_ensemble.py --model configs/model/OmniESI_ensemble.yaml --data configs/data/CatPred_kcat.yaml --weight_folder ./results/CatPred_kcat/fold_OmniESI/ --task regression

echo "Inference CatPred-DB km task"
CUDA_VISIBLE_DEVICES=0 python test_ensemble.py --model configs/model/OmniESI_ensemble.yaml --data configs/data/CatPred_km.yaml --weight_folder ./results/CatPred_km/fold_OmniESI/ --task regression

echo "Inference CatPred-DB ki task"
CUDA_VISIBLE_DEVICES=0 python test_ensemble.py --model configs/model/OmniESI_ensemble.yaml --data configs/data/CatPred_ki.yaml --weight_folder ./results/CatPred_ki/fold_OmniESI/ --task regression



# ESP task
echo "Inference ESP task"
CUDA_VISIBLE_DEVICES=0 python test.py --model configs/model/OmniESI.yaml --data configs/data/esp.yaml --weight ./results/esp/OmniESI/best_model_epoch.pth --task binary



# Mutational effect tasks
echo "Inference epistasis_amp task"
CUDA_VISIBLE_DEVICES=0 python test.py --model configs/model/OmniESI.yaml --data configs/data/epistasis_amp.yaml --weight ./results/epistasis_amp/OmniESI/best_model_epoch.pth --task binary

echo "Inference epistasis_amp task"
CUDA_VISIBLE_DEVICES=0 python test.py --model configs/model/OmniESI.yaml --data configs/data/epistasis_ctx.yaml --weight ./results/epistasis_ctx/OmniESI/best_model_epoch.pth --task binary

echo "Inference mut_classify task"
CUDA_VISIBLE_DEVICES=0 python test.py --model configs/model/OmniESI.yaml --data configs/data/mut_classify.yaml --weight ./results/mut_classify/OmniESI/best_model_epoch.pth --task binary



# Ablation study on CatPred-DB_kact and kat_km tasks
echo "Ablation study on CatPred-DB_kact"
CUDA_VISIBLE_DEVICES=0 python test_ensemble.py --model configs/model/Baseline_ensemble.yaml --data configs/data/CatPred_kcat.yaml --weight_folder ./results/CatPred_kcat/fold_Baseline/ --task regression
CUDA_VISIBLE_DEVICES=0 python test_ensemble.py --model configs/model/BCFM_ensemble.yaml --data configs/data/CatPred_kcat.yaml --weight_folder ./results/CatPred_kcat/fold_BCFM/ --task regression
CUDA_VISIBLE_DEVICES=0 python test_ensemble.py --model configs/model/CCFM_ensemble.yaml --data configs/data/CatPred_kcat.yaml --weight_folder ./results/CatPred_kcat/fold_CCFM/ --task regression
CUDA_VISIBLE_DEVICES=0 python test_ensemble.py --model configs/model/OmniESI_ensemble.yaml --data configs/data/CatPred_kcat.yaml --weight_folder ./results/CatPred_kcat/fold_OmniESI/ --task regression

echo "Ablation study on kat_km"
CUDA_VISIBLE_DEVICES=0 python test.py --model configs/model/Baseline.yaml --data configs/data/kcat_km.yaml --weight ./results/kcat_km/Baseline/best_model_epoch.pth --task regression
CUDA_VISIBLE_DEVICES=0 python test.py --model configs/model/BCFM.yaml --data configs/data/kcat_km.yaml --weight ./results/kcat_km/BCFM/best_model_epoch.pth --task regression
CUDA_VISIBLE_DEVICES=0 python test.py --model configs/model/CCFM.yaml --data configs/data/kcat_km.yaml --weight ./results/kcat_km/CCFM/best_model_epoch.pth --task regression
CUDA_VISIBLE_DEVICES=0 python test.py --model configs/model/OmniESI.yaml --data configs/data/kcat_km.yaml --weight ./results/kcat_km/OmniESI/best_model_epoch.pth --task regression
