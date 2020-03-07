import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNClassifier(nn.Module):
    def __init__(self, gnn, input_dim, hidden_dim, num_labels, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(gnn(input_dim, hidden_dim, F.relu))
            elif i == num_layers - 1:
                layers.append(gnn(hidden_dim, num_labels))
            else:
                layers.append(gnn(hidden_dim, hidden_dim, F.relu))
        self.gnn_layers = nn.ModuleList(layers)

    def forward(self, g, nodes):
        x = nodes
        for layer in self.gnn_layers:
            x = layer(g, x)
        return x