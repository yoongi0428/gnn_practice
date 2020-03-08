import torch
import torch.nn as nn

from GNNs.classifier import GNNClassifier

class GCN(GNNClassifier):
    def __init__(self, input_dim, hidden_dim, num_labels, num_layers):
        super().__init__(input_dim, hidden_dim, num_labels, num_layers)

        self.build_layers()

    def build_layers(self):
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(GCNLayer(self.input_dim, self.hidden_dim, F.tanh))
            elif i == self.num_layers - 1:
                layers.append(GCNLayer(self.hidden_dim, self.num_labels))
            else:
                layers.append(GCNLayer(self.hidden_dim, self.hidden_dim, F.tanh))
        self.gnn_layers = nn.ModuleList(layers)

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = activation

    def message_func(self, edges):
        return {'m': edges.src['h']}

    def reduce_func(self, nodes):
        return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

    def apply_nodes(self, nodes):
        h = self.linear(nodes.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h': h}

    def forward(self, g, h):
        g.ndata['h'] = h
        g.update_all(self.message_func, self.reduce_func)
        g.apply_nodes(func=self.apply_nodes)
        return g.ndata.pop('h')