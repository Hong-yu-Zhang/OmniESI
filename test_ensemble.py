from models import OmniESI
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import ESIDataset
from torch.utils.data import DataLoader
import torch
import argparse
import warnings, os
import pandas as pd
from run.tester import Tester
from run.tester_reg import Tester_Reg
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_ood_indices(train_clusters, test_clusters):
    """
    Adapt from CatPred
    """
    return [i for i, cluster in enumerate(test_clusters) if cluster not in train_clusters]


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
parser = argparse.ArgumentParser(description="OmniESI for multi-purpose ESI prediction [TEST]")
parser.add_argument('--model', required=True, help="path to model config file", type=str)
parser.add_argument('--data', required=True, help="path to data config file", type=str)
parser.add_argument('--weight_folder', required=True, help="path to model weight", type=str)
parser.add_argument('--split', default='test', type=str, help="specify which folder as test set", choices=['test', 'val'])
parser.add_argument('--task', required=True, help="task type: regression, binary", choices=['regression', 'binary'], type=str)
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    
    cfg.merge_from_file(args.model)
    cfg.merge_from_file(args.data)
    set_seed(42)
    
    print(f"Model Config: {args.model}")
    print(f"Data Config: {args.data}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = cfg.SOLVER.DATA

    test_path = os.path.join(dataFolder, f"{args.split}.csv")

    df_test = pd.read_csv(test_path)
    test_dataset = ESIDataset(df_test.index.values, df_test, args.task)
    
    
    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': False, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': False, 'collate_fn': graph_collate_func}
    
    test_generator = DataLoader(test_dataset, **params)

    seed_list = cfg.SOLVER.SEED

    y_pred_ensemble = []
    model_name = os.path.basename(args.model).split('.')[0]
    for seed in seed_list:
        model = OmniESI(**cfg)

        weight_path = os.path.join(args.weight_folder, f"{model_name}_{seed}/best_model_epoch.pth")

        torch.backends.cudnn.benchmark = True
        
        if args.task == 'binary':
            tester = Tester(model, device, test_generator, weight_path, **cfg)
        else:
            tester = Tester_Reg(model, device, test_generator, weight_path, **cfg)

        _, y_pred, y_label = tester.test()

        y_pred = np.array(y_pred)
        y_label = np.array(y_label)

        y_pred_ensemble.append(y_pred)

    print(f'Parameters: {count_parameters(model)}')

    y_pred = np.mean(y_pred_ensemble, axis=0)

    y_pred = y_pred.flatten()
    y_label = y_label.flatten()
    
    pcc, _ = pearsonr(y_label, y_pred)
    rmse = np.sqrt(mean_squared_error(y_label, y_pred))
    mae = mean_absolute_error(y_label, y_pred)
    r2 = r2_score(y_label, y_pred)
    
    print(f"======> For Threhold 100")
    print("PCC:", pcc)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)


    # OOD performance for different cluster levels
    for N in [99, 80, 60, 40]:
        train_clusters = set(df_train[f'sequence_{N}cluster'])
        test_clusters = df_test[f'sequence_{N}cluster']

        OOD_INDICES = get_ood_indices(train_clusters, test_clusters)

        y_label_ood = y_label[OOD_INDICES]
        y_pred_ood = y_pred[OOD_INDICES]
        
        pcc, _ = pearsonr(y_label_ood, y_pred_ood)
        rmse = np.sqrt(mean_squared_error(y_label_ood, y_pred_ood))
        mae = mean_absolute_error(y_label_ood, y_pred_ood)
        r2 = r2_score(y_label_ood, y_pred_ood)
        
        print(f"======> For Threhold {N}")
        print(f"Sample Num: {len(y_label_ood)}")
        print("PCC:", pcc)
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("R2:", r2)

    return None


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
