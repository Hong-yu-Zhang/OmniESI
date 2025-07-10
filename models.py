import torch.nn as nn
import torch.nn.functional as F
import torch
from module.Encoder import *
from module.CN import *

class OmniESI(nn.Module):
    def __init__(self, **config):
        super(OmniESI, self).__init__()
        """
        drug: features related to substrate;
        protein: features related to enzyme;     
        """
        
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        drug_padding = config["DRUG"]["PADDING"]

        protein_in_dim = config["PROTEIN"]["IN_DIM"]
        protein_hidden_dim = config["PROTEIN"]["HIDDEN_DIM"]
        protein_target_dim = config["PROTEIN"]["TARGET_DIM"]
        
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        out_binary = config["DECODER"]["BINARY"]
        
        self.stage_num = config["STAGE"]["NUM"]
        self.ccfm_flag = config["STAGE"]["CCFM"]
        self.bcfm_flag = config["STAGE"]["BCFM"]
        
        self.ccfm_dim = config["CCFM"]["DIM"]    
        self.bcfm_dim = config["BCFM"]["DIM"]
        
        self.drug_extractor = Encoder_drug(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)        
        self.protein_extractor = Encoder_protein(protein_in_dim, protein_hidden_dim, protein_target_dim)
        
        if self.bcfm_flag:
            self.bcfm_list = nn.ModuleList([BCFM(dim_model=self.bcfm_dim) for i in range(self.stage_num)])
    
        if self.ccfm_flag:
            self.fusion = CCFM(dim_model=self.ccfm_dim)
        else:
            self.fusion = SimpleFusion()
            
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, v_d, v_p, v_d_mask, v_p_mask):
        v_d = self.drug_extractor(v_d)
        v_p = self.protein_extractor(v_p)
        
        if self.bcfm_flag:
            for i in range(self.stage_num):
                v_p, v_d = self.bcfm_list[i](v_p, v_d, v_p_mask, v_d_mask)
        if self.ccfm_flag:
            f = self.fusion(v_p, v_d, v_p_mask, v_d_mask)
        else:
            f = self.fusion(v_d, v_p, v_d_mask, v_p_mask)
        
        score = self.mlp_classifier(f)

        return v_d, v_p, f, score

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Tanh(),
        )
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent

class SimpleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = MaskedAveragePooling()

    def forward(self, v_d, v_p, v_d_mask, v_p_mask):
        tensor1_pooled = self.avgpool(v_d, v_d_mask)
        tensor2_pooled = self.avgpool(v_p, v_p_mask)

        concatenated = tensor1_pooled + tensor2_pooled

        return concatenated