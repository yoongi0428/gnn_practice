import os
import yaml
import torch
import numpy as np
import networkx as nx
from dgl.data import citation_graph
from dgl import DGLGraph

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from GNNs import GCN, GAT

def set_random_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_cora_data(device):
    data = citation_graph.load_cora()
    print(data)
    
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

def build_classifier(gnn, input_dim, hidden_dim, num_labels, num_layers):
    file = open('GNN_config.yml', 'r')
    # cfg = yaml.load(file, Loader=yaml.FullLoader)
    cfg = yaml.load(file)

    if gnn == 'GCN':
        return GCN(input_dim, hidden_dim, num_labels, num_layers)
    elif gnn == 'GAT':
        return GAT(input_dim, hidden_dim, num_labels, num_layers, **cfg[gnn])
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

def visualize_logits(g, all_logits):
    colors = ['#C7980A', '#F4651F', '#82D8A7', '#CC3A05', '#575E76', '#156943', '#0BD055', '#ACD338']
    nx_g = g.to_networkx().to_undirected()
    num_nodes = all_logits[0].shape[0]
    def draw(i):
        pos = {}
        colors = []
        for v in range(num_nodes):
            pos[v] = all_logits[i][v].numpy()
            pred = pos[v].argmax()
            colors.append(colors[pred])
        ax.cla()
        ax.axis('off')
        ax.set_title('Epoch: %d' % i)
        nx.draw_networkx(nx_g, pos, node_color=colors,
                with_labels=True, node_size=300, ax=ax)

    fig = plt.figure(dpi=150)
    fig.clf()
    ax = fig.subplots()
    ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)
    plt.close() 