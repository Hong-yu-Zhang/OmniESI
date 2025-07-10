from models import OmniESI
from time import time
from utils import set_seed
from configs import get_cfg_defaults

import argparse
import warnings, os
import pandas as pd
from tqdm import tqdm
from functools import partial


import torch
import dgl
from scripts.embedding import ESM_model
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
parser = argparse.ArgumentParser(description="OmniESI for multi-purpose ESI prediction [TEST]")
parser.add_argument('--model', required=True, help="path to model config file", type=str)
parser.add_argument('--csv_data', required=True, help="path to inference data", type=str)
parser.add_argument('--weight', required=True, help="path to model weight", type=str)
parser.add_argument('--task', required=True, help="task type: regression, binary", choices=['regression', 'binary'], type=str)

args = parser.parse_args()


def prepare_inputs(smiles, protein_seq, embedder):
    # 1. SMILES to graph
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
    fc = partial(smiles_to_bigraph, add_self_loop=True)
    v_d = fc(smiles=smiles, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)

    drug = v_d
    actual_node_feats = drug.ndata.pop('h')
    num_actual_nodes = actual_node_feats.shape[0]
    num_virtual_nodes = 0
    virtual_node_bit = torch.zeros([num_actual_nodes, 1])
    actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
    drug.ndata['h'] = actual_node_feats
    virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
    drug.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
    drug = drug.add_self_loop()
    v_d = dgl.batch([drug])
    v_d_mask = torch.zeros(num_actual_nodes, dtype=torch.bool).unsqueeze(0)

    # 2. Protein sequence to embedding
    v_p = embedder([protein_seq])
    v_p = v_p[:, 1:len(protein_seq)+1, :]
    v_p_mask = torch.zeros(v_p.shape[1], dtype=torch.bool).unsqueeze(0)

    return v_d, v_p, v_d_mask, v_p_mask

def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    
    cfg.merge_from_file(args.model)
    set_seed(42)
    
    print(f"Model Config: {args.model}")
    print(f"Running on: {device}", end="\n\n")

    model = OmniESI(**cfg)
    weight_path = args.weight
    model.load_state_dict(torch.load(weight_path))
    model.to(device)

    esm_model = ESM_model()
    esm_model.device = device

    torch.backends.cudnn.benchmark = True

    df = pd.read_csv(args.csv_data)

    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for index, row in tqdm(df.iterrows()):
            smiles = row['SMILES']
            protein_seq = row['Protein']
            v_d, v_p, v_d_mask, v_p_mask = prepare_inputs(smiles, protein_seq, esm_model)
            v_d, v_p, v_d_mask, v_p_mask = v_d.to(device), v_p.to(device), v_d_mask.to(device), v_p_mask.to(device)
            output = model(v_d, v_p, v_d_mask, v_p_mask)
            y = output[-1]
            if args.task == 'regression':
                y_pred_list.append(y)
            else:
                y_pred_list.append(y.sigmoid())
    
    y_pred_list = torch.cat(y_pred_list, dim=0)
    y_pred_list = y_pred_list.cpu().numpy()
    y_pred_list = y_pred_list.tolist()

    csv_file_name = os.path.basename(args.csv_data)
    csv_file_name = csv_file_name.split('.')[0]
    df['OmniESI_pred'] = y_pred_list
    df.to_csv(f'OmniESI_pred_{csv_file_name}.csv', index=False)

    return 0


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
