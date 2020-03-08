import os
import yaml
import torch
import numpy as np
import networkx as nx
from dgl.data import citation_graph
from dgl import DGLGraph

import matplotlib.animation as animation
import matplotlib.pyplot as plt

def set_random_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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