import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from models import binary_cross_entropy, cross_entropy_logits
from prettytable import PrettyTable
from tqdm import tqdm


class Tester_Reg(object):
    def __init__(self, model, device, test_dataloader, weight_path, alpha=1, **config):
        self.weight_path = weight_path
        self.model = model
        self.device = device
        self.test_dataloader = test_dataloader
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.step = 0

        self.test_metrics = {}
        self.config = config
        self.loss_func = nn.MSELoss()

    def test(self, dataloader="test"):
        float2str = lambda x: '%0.4f' % x
        test_loss = 0
        feat_list = []
        y_label, y_pred = [], []
        data_loader = self.test_dataloader
        num_batches = len(data_loader)
        self.model.load_state_dict(torch.load(self.weight_path))
        self.model.to(self.device)
        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, labels, v_d_mask, v_p_mask) in enumerate(tqdm(data_loader)):
                v_d, v_p, labels, v_d_mask, v_p_mask = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device), v_d_mask.to(self.device), v_p_mask.to(self.device)
                v_d, v_p, f, score = self.model(v_d, v_p, v_d_mask, v_p_mask)
                loss = self.loss_func(score, labels.unsqueeze(-1))
                test_loss += loss.item()
                feat_list = feat_list + f.to("cpu").tolist()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + score.to("cpu").tolist()
        
        test_loss = test_loss / num_batches
        pcc, _ = pearsonr(np.array(y_label).flatten(), np.array(y_pred).flatten())
        rmse = np.sqrt(mean_squared_error(y_label, y_pred))
        mae = mean_absolute_error(y_label, y_pred)
        r2 = r2_score(y_label, y_pred)

        # Print out the metrics
        print("MSE Loss", test_loss)
        print("PCC:", pcc)
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("R2:", r2)
        
        return test_loss, y_pred, y_label
