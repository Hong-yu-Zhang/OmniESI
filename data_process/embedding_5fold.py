# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import argparse

from torch import nn
import torch.nn.functional as F
import esm
import os
from tqdm import tqdm

class ESM_model(nn.Module):
    def __init__(self, ):
        super(ESM_model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model = self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()
    def forward(self, data):
        tmp = []
        for seq in data:
            tmp.append(('',seq))
        data = tmp
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens[:, 0:1024]
        
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        return token_representations

parser = argparse.ArgumentParser(description="Embedding extraction")
parser.add_argument('--feat_dir', required=True, help="path to embeddings", type=str)
args = parser.parse_args()

if __name__ == '__main__':
    dataset_list = ['DLKcat', 'GraphKM', 'MPEK_kcat', 'MPEK_km']
    for dataset in dataset_list:
        print(f'Embedding extraction for: {dataset}')
        path = f'../datasets/{dataset}/'
        
        df = pd.read_csv(os.path.join(path, f'{dataset}.csv'))
        protein_set = df['Protein'].drop_duplicates().reset_index(drop=True)
        protein_index = {protein: index for index, protein in enumerate(protein_set)}
        df['Protein_index'] = df['Protein'].map(protein_index)
    
        model = ESM_model()
    
        dataset_dir = f'{args.feat_dir}/{dataset}'
        feat_dir = os.path.join(dataset_dir, 'esm')
    
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            os.makedirs(feat_dir)
    
        for seq, index in tqdm(protein_index.items()):
            feat = model([seq])
            feat = feat.squeeze(dim=0)
            feat = feat[1:len(seq)+1, :]
            feat = feat.cpu().detach()
            torch.save(feat, os.path.join(feat_dir, f'{index}.pt'))
            
        df['Protein_index'] = df['Protein'].map(protein_index)
        df['Protein_Path'] = df['Protein_index'].apply(lambda x: f'{dataset_dir}/esm/{x}.pt')
        df.to_csv(os.path.join(path, f'{dataset}_650m.csv'), index=False)
