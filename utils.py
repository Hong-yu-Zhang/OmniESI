import os
import random
import numpy as np
import torch
import dgl
import logging

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25


def set_seed(seed=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.enabled = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
        

def graph_collate_func(x):
    d, p, y = zip(*x)
    max_drug_lenth = max(i.num_nodes() for i in d)
    
    max_protein_lenth = max(i.size(0) for i in p)

    padded_protein_feat = []
    protein_masks = []  # 用于存储每个蛋白质的mask

    for feat in p:
        num_protein_pad = max_protein_lenth - feat.size(0)
        mask = torch.ones(max_protein_lenth, dtype=torch.bool)

        if num_protein_pad > 0:
            feat = torch.nn.functional.pad(feat, (0, 0, 0, num_protein_pad), value=0)
            mask[:-num_protein_pad] = 0  # 标记填充的部分为True
        else:
            mask = ~mask

        feat = feat.unsqueeze(0)
        padded_protein_feat.append(feat)
        protein_masks.append(mask)

    padding_protein_mask = torch.stack(protein_masks)

    drug_feat = []
    node_counts = []
    for drug in d:
        actual_node_feats = drug.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        node_counts.append(num_actual_nodes)  # 记录实际节点数
        num_virtual_nodes = max_drug_lenth - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        drug.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        drug.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        drug = drug.add_self_loop()
        drug_feat.append(drug)
    drug_feat = dgl.batch(drug_feat)

    padding_drug_mask = torch.ones(len(node_counts), max_drug_lenth, dtype=torch.bool)
    for i, count in enumerate(node_counts):
        padding_drug_mask[i, :count] = 0
    return drug_feat, torch.cat(padded_protein_feat, dim=0), torch.tensor(y), padding_drug_mask, padding_protein_mask


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def integer_label_protein(sequence):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(len(sequence))
    for idx, letter in enumerate(sequence):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding
