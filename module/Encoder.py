import torch.nn as nn
import torch
from dgllife.model.gnn import GCN


class Encoder_drug(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(Encoder_drug, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]
    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

"""
ESM Feature For Protein 
"""
class Encoder_protein(nn.Module):
    def __init__(self, embedding_dim=320, hidden_dim=320, target_dim=128):
        super(Encoder_protein, self).__init__()
        self.output_layer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x):
        x = self.output_layer(x)
        return x