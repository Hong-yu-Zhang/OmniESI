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
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def calculate_ood_metrics(df, threshold_ranges, y_label_col="y_label", y_pred_col="y_pred"):
    """
    Calculate OOD metrics (PCC, RMSE, MAE, R2) over specific similarity threshold ranges.
    Args:
        df (DataFrame): The dataframe containing the test results.
        threshold_ranges (list): List of threshold ranges as pairs (min_threshold, max_threshold).
        y_label_col (str): The column name for true labels.
        y_pred_col (str): The column name for predicted values.
    """
    for threshold_range in threshold_ranges:
        min_threshold, max_threshold = threshold_range
        print(f"======> For Threshold Range [{min_threshold}, {max_threshold}]")
        
        filtered_df = df[(df["similarity"] > min_threshold) & (df["similarity"] <= max_threshold)]
        
        if len(filtered_df) == 0:
            print(f"No data points in this threshold range [{min_threshold}, {max_threshold}]")
            continue
        
        y_label = filtered_df[y_label_col].values
        y_pred = filtered_df[y_pred_col].values

        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        
        
        y_pred_s = [1 if i > 0.5 else 0 for i in y_pred]
        
        cm1 = confusion_matrix(y_label, y_pred_s)
        accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
        mcc = matthews_corrcoef(y_label, y_pred_s)
        # 计算精确率，召回率和 F1 分数
        precision = precision_score(y_label, y_pred_s)
        recall = recall_score(y_label, y_pred_s)
        f1 = f1_score(y_label, y_pred_s)

        print("Precision:", precision)
        print("Recall:", recall)
        print("Accuracy", accuracy)
        print("AUROC:", auroc)
        print("AURPC:", auprc)
        print("F1 Score:", f1)
        print("MCC:", mcc)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
parser = argparse.ArgumentParser(description="OmniESI for multi-purpose ESI prediction [TEST]")
parser.add_argument('--model', required=True, help="path to model config file", type=str)
parser.add_argument('--data', required=True, help="path to data config file", type=str)
parser.add_argument('--weight', required=True, help="path to model weight", type=str)
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


    model = OmniESI(**cfg)

    weight_path = args.weight

    torch.backends.cudnn.benchmark = True
    
    if args.task == 'binary':
        tester = Tester(model, device, test_generator, weight_path, **cfg)
    else:
        tester = Tester_Reg(model, device, test_generator, weight_path, **cfg)

    _, y_pred, y_label = tester.test()

    df_test["y_pred"] = y_pred
    df_test["y_label"] = y_label

    print(f'Parameters: {count_parameters(model)}')

    if 'similarity' in df_test.columns:
        threshold_ranges = [[0, 40], [40, 60], [60, 80]]  # for ESP
        calculate_ood_metrics(df_test, threshold_ranges)
    else:
        print("No OOD Annotation")

    return None


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
