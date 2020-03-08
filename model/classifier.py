import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_layers = num_layers

    def build_layers(self):
        raise NotImplementedError
    
    def forward(self, g, nodes):
        x = nodes
        for layer in self.gnn_layers:
            x = layer(g, x)
        return x