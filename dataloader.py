import torch.utils.data as data
import torch
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein
from tqdm import tqdm
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