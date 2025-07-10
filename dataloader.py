import torch.utils.data as data
import torch
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
"""
Load feature from hard disk.
"""
class ESIDataset(data.Dataset):
    def __init__(self, list_IDs, df, task='binary'):
        self.list_IDs = list_IDs
        self.df = df
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        self.task = task

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['SMILES']
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)

        v_p = torch.load(self.df.iloc[index]['Protein_Path'])

        if self.task == 'binary':
            y = self.df.iloc[index]["Y"]
        else:
            y = self.df.iloc[index]["Score"]      
        return v_d, v_p, y
    
"""
Load feature from Memory.
"""
# class ESIDataset(data.Dataset):
#     def __init__(self, list_IDs, df, task='binary'):
#         self.list_IDs = list_IDs
#         self.df = df
#         self.atom_featurizer = CanonicalAtomFeaturizer()
#         self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
#         self.fc = partial(smiles_to_bigraph, add_self_loop=True)
#         self.task = task

#         with ThreadPoolExecutor(max_workers=32) as executor:
#             print("Load ESM Feature in Memory")
#             v_p_paths = df['Protein_Path'].tolist()
#             # 多线程加载 v_p 数据
#             self.v_p_data = list(tqdm(executor.map(torch.load, v_p_paths), total=len(v_p_paths), desc="Loading ESM data"))

#     def __len__(self):
#         return len(self.list_IDs)

#     def __getitem__(self, index):
#         index = self.list_IDs[index]
#         v_d = self.df.iloc[index]['SMILES']
#         v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)

#         v_p = self.v_p_data[index]

#         if self.task == 'binary':
#             y = self.df.iloc[index]["Y"]
#         else:
#             y = self.df.iloc[index]["Score"]      
#         return v_d, v_p, y