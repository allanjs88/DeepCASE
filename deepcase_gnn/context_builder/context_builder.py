"""Jointly learned host-local GRU and directed mean-aggregation GraphSAGE."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.nn import SAGEConv


class ContextBuilder(nn.Module):
    def __init__(self, features, hidden_size=64, embedding_size=32, gnn_layers=2, dropout=0.1):
        super().__init__()
        if features < 3 or hidden_size < 1 or embedding_size < 1 or gnn_layers < 0:
            raise ValueError("Invalid vocabulary, hidden dimensions, or GNN layer count.")
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(features, embedding_size, padding_idx=0)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.node_projection = nn.Linear(hidden_size + 4, hidden_size)
        self.convs = nn.ModuleList([SAGEConv(hidden_size, hidden_size, aggr="mean")
                                    for _ in range(gnn_layers)])
        self.dropout = nn.Dropout(dropout)
        # Both ordered endpoint representations and historical pair statistics.
        # Output index 0 represents UNK; padding is never a prediction target.
        self.event_head = nn.Sequential(nn.Linear(2 * hidden_size + 4, hidden_size),
                                        nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(hidden_size, features - 1))

    def encode(self, graph):
        embedded = self.embedding(graph.sequence)
        packed = pack_padded_sequence(embedded, graph.lengths.clamp_min(1).cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        # An empty sequence has exactly zero GRU features, including unseen hosts.
        temporal = hidden[-1] * graph.lengths.ne(0).unsqueeze(1)
        nodes = torch.relu(self.node_projection(torch.cat([temporal, graph.x], dim=1)))
        for conv in self.convs:
            nodes = self.dropout(torch.relu(conv(nodes, graph.edge_index)))
        return nodes

    def forward(self, graph):
        nodes = self.encode(graph)
        source, destination = graph.target_index
        return self.event_head(torch.cat([nodes[source], nodes[destination], graph.target_attr], dim=1))
