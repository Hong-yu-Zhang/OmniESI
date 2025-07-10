import torch
import torch.nn as nn
import copy
import os
import numpy as np
from utils import set_seed, graph_collate_func, mkdir
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from prettytable import PrettyTable
from tqdm import tqdm


class Trainer_Reg(object):
    def __init__(self, seed, model, optim, device, train_dataloader, val_dataloader, test_dataloader, output, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.alpha = alpha
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0

        self.seed = seed

        self.best_model = None
        self.best_epoch = None
        self.best_rmse = 100
        self.best_r2 = -1

        self.loss_func = nn.MSELoss()

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.val_loss_epoch, self.val_rmse_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = output

        valid_metric_header = ["# Epoch", "PCC", "RMSE", "MAE", "R2", "Val_loss"]
        test_metric_header = ["# Best Epoch", "PCC", "RMSE", "MAE", "R2", "Test_loss"]
        
        train_metric_header = ["# Epoch", "Train_loss"]
        
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

    def train(self):
        set_seed(self.seed)
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            
            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            val_loss, mae, rmse, pcc, r2 = self.test(dataloader="val")
        
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [pcc, rmse, mae, r2, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_rmse_epoch.append(rmse)
            if rmse <= self.best_rmse and r2 >= self.best_r2:
                self.best_model = copy.deepcopy(self.model)
                self.best_rmse = rmse
                self.best_r2 = r2
                self.best_epoch = self.current_epoch
                torch.save(self.best_model.state_dict(), os.path.join(self.output_dir, f"best_model_epoch.pth"))      
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " RMSE "
                  + str(rmse) + " R2 " + str(r2))
        test_loss, mae, rmse, pcc, r2 = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [pcc, rmse, mae, r2, test_loss]))
        self.test_table.add_row(test_lst)
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " RMSE "
              + str(rmse) + "PCC" + str(pcc) + " MAE " + str(mae) + " R2 " +
              str(r2))
        self.test_metrics["pcc"] = pcc
        self.test_metrics["rmse"] = rmse
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["mae"] = mae
        self.test_metrics["pcc"] = pcc
        self.test_metrics["best_epoch"] = self.best_epoch
        self.save_result()
    
        return self.test_metrics

    def save_result(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        torch.save(self.best_model.state_dict(),
                   os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
       
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (v_d, v_p, labels, v_d_mask, v_p_mask) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d, v_p, labels, v_d_mask, v_p_mask = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device), v_d_mask.to(self.device), v_p_mask.to(self.device)
            self.optim.zero_grad()
            v_d, v_p, f, score= self.model(v_d, v_p, v_d_mask, v_p_mask)     
            loss = self.loss_func(score.squeeze(-1), labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            
        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch

    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, labels, v_d_mask, v_p_mask) in enumerate(tqdm(data_loader)):
                v_d, v_p, labels, v_d_mask, v_p_mask = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device), v_d_mask.to(self.device), v_p_mask.to(self.device)
                if dataloader == "val":
                    v_d, v_p, f, score = self.model(v_d, v_p, v_d_mask, v_p_mask)
                elif dataloader == "test":
                    v_d, v_p, f, score = self.best_model(v_d, v_p, v_d_mask, v_p_mask)
                loss = self.loss_func(score.squeeze(), labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + score.to("cpu").tolist()
        test_loss = test_loss / num_batches
        pcc, _ = pearsonr(y_label, y_pred)
        rmse = np.sqrt(mean_squared_error(y_label, y_pred))
        mae = mean_absolute_error(y_label, y_pred)
        r2 = r2_score(y_label, y_pred)
        return test_loss, mae, rmse, pcc, r2