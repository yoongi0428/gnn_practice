import os
import torch
import numpy as np
import networkx as nx
from dgl.data import citation_graph
from dgl import DGLGraph

from model import GCN, GAT

def load_cora_data(device):
    data = citation_graph.load_cora()

    features = torch.FloatTensor(data.features).to(device)
    labels = torch.LongTensor(data.labels).to(device)
    train_mask = torch.BoolTensor(data.train_mask).to(device)
    valid_mask = torch.BoolTensor(data.val_mask).to(device)
    test_mask = torch.BoolTensor(data.test_mask).to(device)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, valid_mask, test_mask

def build_classifier(gnn, input_dim, hidden_dim, num_labels, args):
    if gnn == 'GCN':
        return GCN(input_dim, hidden_dim, num_labels, args.num_layers)
    elif gnn == 'GAT':
        return GAT(input_dim, hidden_dim, num_labels, args.num_layers, args.num_heads, args.merge, args.dropout)
    else:
        raise NotImplementedError('%d is not implemented yet or doesn\'t exist.' % gnn)

def evaluate(classifier, g, features, labels, mask):
    classifier.eval()

    logits = classifier(g, features)

    pred = torch.argmax(logits[mask], 1)
    labels = labels[mask]

    num_eval = len(pred)
    num_correct = torch.sum(pred == labels).item()

    acc = num_correct / num_eval
    return acc