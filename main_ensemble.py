from models import OmniESI
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import ESIDataset
from torch.utils.data import DataLoader
from run.trainer import Trainer
from run.trainer_DDP import Trainer_DDP
from run.trainer_reg import Trainer_Reg
from run.trainer_reg_DDP import Trainer_Reg_DDP
import torch
import argparse
import warnings, os, re
import pandas as pd
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np

def create_experiment_directory(result, dataset):
    # DDP initialized; is main process?
    if dist.is_initialized():
        rank = dist.get_rank()
        is_main_process = (rank == 0)
    else:
        # DDP not initialized
        is_main_process = True

    base_path = os.path.join(result, dataset)
    new_exp_path = ""
    if not os.path.exists(base_path):
        try:
            os.makedirs(base_path)
        except FileExistsError:
            pass
            
    if dist.is_initialized():
        dist.barrier()

    exp_folders = [d for d in os.listdir(base_path) if d.startswith('exp') and os.path.isdir(os.path.join(base_path, d))]
    exp_numbers = [int(re.search(r'exp(\d+)', folder).group(1)) for folder in exp_folders if re.search(r'exp(\d+)', folder)]
    max_exp_number = max(exp_numbers) if exp_numbers else -1
    new_exp_number = max_exp_number + 1
    new_exp_folder = f'exp{new_exp_number}'
    new_exp_path = os.path.join(base_path, new_exp_folder)

    if is_main_process:
        if not os.path.exists(new_exp_path):
            os.makedirs(new_exp_path)
        os.system(f'cp -r ./module {new_exp_path}/')
        os.system(f'cp ./models.py {new_exp_path}/')

    if dist.is_initialized():
        dist.barrier()
    return new_exp_path

if torch.cuda.device_count() > 1:
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
parser = argparse.ArgumentParser(description="OmniESI for multi-purpose ESI prediction [TRAIN]")
parser.add_argument('--model', required=True, help="path to model config file", type=str)
parser.add_argument('--data', required=True, help="path to data config file", type=str)
parser.add_argument('--task', required=True, help="task type: regression, binary", choices=['regression', 'binary'], type=str)
args = parser.parse_args()

def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.model)
    cfg.merge_from_file(args.data)
    
    print(f"Model Config: {args.model}")
    print(f"Data Config: {args.data}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = cfg.SOLVER.DATA

    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, "val.csv")
    test_path = os.path.join(dataFolder, "test.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    train_dataset = ESIDataset(df_train.index.values, df_train, args.task)
    val_dataset = ESIDataset(df_val.index.values, df_val, args.task)
    test_dataset = ESIDataset(df_test.index.values, df_test, args.task)
    
    if torch.cuda.device_count() > 1:
        params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': False, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                  'drop_last': True, 'collate_fn': graph_collate_func}
    
        training_generator = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset), **params)
        params['drop_last'] = False
        
        val_generator = DataLoader(val_dataset, sampler=DistributedSampler(val_dataset), **params)
        test_generator = DataLoader(test_dataset, sampler=DistributedSampler(test_dataset), **params)
    else:
        params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}

        training_generator = DataLoader(train_dataset, **params)
        params['shuffle'] = False
        params['drop_last'] = False
    
        val_generator = DataLoader(val_dataset, **params)
        test_generator = DataLoader(test_dataset, **params)


    torch.backends.cudnn.benchmark = True

    output_dir = os.path.join(cfg.RESULT.OUTPUT_DIR, cfg.SOLVER.SAVE)
    
    base_dir = os.path.join(cfg.RESULT.OUTPUT_DIR, cfg.SOLVER.SAVE)
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    
    # Initialize Trainer
    # Seed in list for possible ensemble training
    y_pred_ensemble = []
    for seed in cfg.SOLVER.SEED:
        print(f"=====> Start Training for Seed {seed}")
        set_seed(seed)
        
        model = OmniESI(**cfg)
        model.to(device)
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[local_rank],
                                                            output_device=local_rank,
                                                            find_unused_parameters=False)

        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

        model_seed_name = f"{model_name}_{seed}"
        output_dir = os.path.join(base_dir, model_seed_name)

        if dist.get_rank() == 0:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            os.system(f'cp -r ./module {output_dir}/')
            os.system(f'cp ./models.py {output_dir}/')
            os.system(f'cp -r {args.data} {output_dir}/')
            os.system(f'cp -r {args.model} {output_dir}/')

        if torch.cuda.device_count() > 1:
            if args.task == 'binary':
                trainer = Trainer_DDP(seed, model, opt, device, training_generator, val_generator, test_generator, val_sampler=DistributedSampler(val_dataset), test_sampler=DistributedSampler(test_dataset), output=output_dir, **cfg)
            else:
                trainer = Trainer_Reg_DDP(seed, model, opt, device, training_generator, val_generator, test_generator, val_sampler=DistributedSampler(val_dataset), test_sampler=DistributedSampler(test_dataset), output=output_dir, **cfg)
        else:
            if args.task == 'binary':
                trainer = Trainer(seed, model, opt, device, training_generator, val_generator, test_generator, output=output_dir, **cfg)
            else:
                trainer = Trainer_Reg(seed, model, opt, device, training_generator, val_generator, test_generator, output=output_dir, **cfg)

        test_metrics, y_pred, y_label = trainer.train()

        if dist.get_rank() == 0:
            y_pred = np.array(y_pred)
            y_label = np.array(y_label)
            df_test['y_pred'] = y_pred
            df_test['y_label'] = y_label
            df_test.to_csv(os.path.join(output_dir, f"prediction_{seed}.csv"))

        y_pred_ensemble.append(y_pred)

    if dist.get_rank() == 0:
        print(f"=====> Final ensemble result: <=====")
        y_pred = np.mean(y_pred_ensemble, axis=0)

        y_pred = y_pred.flatten()
        y_label = y_label.flatten()
        
        pcc, _ = pearsonr(y_label, y_pred)
        rmse = np.sqrt(mean_squared_error(y_label, y_pred))
        mae = mean_absolute_error(y_label, y_pred)
        r2 = r2_score(y_label, y_pred)
        print("PCC:", pcc)
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("R2:", r2)

        print(f"Directory for saving result: {output_dir}")

    return test_metrics


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
