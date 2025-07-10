import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef
from models import binary_cross_entropy, cross_entropy_logits
from prettytable import PrettyTable
from tqdm import tqdm


class Tester(object):
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

    def test(self, dataloader="test"):
        float2str = lambda x: '%0.4f' % x
        test_loss = 0
        y_label, y_pred = [], []
        data_loader = self.test_dataloader
        num_batches = len(data_loader)
        self.model.load_state_dict(torch.load(self.weight_path))
        #self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.to(self.device)
        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, labels, v_d_mask, v_p_mask) in enumerate(tqdm(data_loader)):
                v_d, v_p, labels, v_d_mask, v_p_mask = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device), v_d_mask.to(self.device), v_p_mask.to(self.device)
                v_d, v_p, f, score = self.model(v_d, v_p, v_d_mask, v_p_mask)
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()

        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches
        
        y_pred_s = [1 if i > 0.5 else 0 for i in y_pred]
        
        cm1 = confusion_matrix(y_label, y_pred_s)
        accuracy = (cm1[0, 0] + cm1[1, 1]) / np.sum(cm1)
        mcc = matthews_corrcoef(y_label, y_pred_s)
        
        # Print out the optimized threshold and metrics
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
        
        return test_loss, y_pred, y_label