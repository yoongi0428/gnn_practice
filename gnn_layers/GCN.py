import torch
import torch.nn as nn

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