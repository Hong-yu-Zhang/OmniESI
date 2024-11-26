from models import MESI
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
parser = argparse.ArgumentParser(description="MESI for multi-purpose ESI prediction [TEST]")
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
    set_seed(cfg.SOLVER.SEED)
    
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


    model = MESI(**cfg)

    weight_path = args.weight

    torch.backends.cudnn.benchmark = True
    
    if args.task == 'binary':
        tester = Tester(model, device, test_generator, weight_path, **cfg)
    else:
        tester = Tester_Reg(model, device, test_generator, weight_path, **cfg)

    result = tester.test()
    print(f'Parameters: {count_parameters(model)}')

    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
