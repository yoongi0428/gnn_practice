import torch
import torch.nn as nn
import torch.nn.functional as F

from model.classifier import GNNClassifier

class GAT(GNNClassifier):
    def __init__(self, input_dim, hidden_dim, num_labels, num_layers, num_heads=2, merge='mean', dropout=0.6):
        super().__init__(input_dim, hidden_dim, num_labels, num_layers)
        self.num_heads = num_heads
        self.merge = merge
        self.dropout = dropout

        self.build_layers()

    def build_layers(self):
        layers = []
        h_dim = self.hidden_dim * self.num_heads if self.merge == 'cat' else self.hidden_dim
        for i in range(self.num_layers):
            if i == 0:
                layers.append(MultiHeadGATLayer(self.input_dim, self.hidden_dim, self.num_heads, self.merge, self.dropout))
            elif i == self.num_layers - 1:
                layers.append(MultiHeadGATLayer(h_dim, self.num_labels, 1, self.merge, self.dropout))
            else:
                layers.append(MultiHeadGATLayer(h_dim, self.hidden_dim, self.num_heads, self.merge, self.dropout))
        self.gnn_layers = nn.ModuleList(layers)

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=2, merge='mean', dropout=0.5):
        super(MultiHeadGATLayer, self).__init__()
        self.merge = merge
        self.dropout = dropout

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, self.dropout))

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs), 0)

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.6):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_linear = nn.Linear(out_dim * 2, 1, bias=False)
        self.dropout = dropout

    def edge_attention(self, edges):
        cat = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_linear(cat)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(F.softmax(nodes.mailbox['e'], dim=1), self.dropout, self.training)

        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = self.linear(F.dropout(h, self.dropout, self.training))
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')