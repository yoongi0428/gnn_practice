import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from dgl.data import citation_graph as citegrh
from dgl import DGLGraph

from utils import set_random_seed, load_cora_data, build_classifier, evaluate, visualize_logits

parser = argparse.ArgumentParser()
parser.add_argument('--gnn', type=str, default='GAT', choices=['GCN', 'GAT'])
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--early_stop', type=int, default=0)
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--seed', type=int, default=2020)

# GCN
# GAT
args = parser.parse_args()

set_random_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

g, features, labels, train_mask, valid_mask, test_mask = load_cora_data(device)

gnn = args.gnn
num_layers = args.num_layers
num_nodes, input_dim = features.shape
hidden_dim = args.hidden_dim
num_labels = int(labels.max() + 1)

num_epochs = args.num_epochs
lr = args.lr
early_stop = args.early_stop
patience = 0
best_acc = -1
best_epoch = -1

classifier = build_classifier(gnn, input_dim, hidden_dim, num_labels, num_layers).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

all_logits = []
for epoch in range(1, args.num_epochs + 1):
    classifier.train()
    optimizer.zero_grad()

    logits = classifier(g, features)
    probs = F.log_softmax(logits, 1)
    loss = F.nll_loss(probs[train_mask], labels[train_mask])

    loss.backward()

    optimizer.step()

    acc = evaluate(classifier, g, features, labels, valid_mask)
    print('epoch %3d, loss %.3f, valid acc %.3f' % (epoch, loss, acc))

    all_logits.append(logits.detach())

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        patience = 0
        
        torch.save(classifier.state_dict(), os.path.join('best_model', '%s_best.pt' % gnn))
    else:
        patience += 1
        if early_stop > 0 and patience >= early_stop:
            print('early stop triggered at epoch %d!\n' % epoch)
            break
# Test model
classifier.load_state_dict(torch.load(os.path.join('best_model', '%s_best.pt' % gnn)))
valid_acc = evaluate(classifier, g, features, labels, valid_mask)
test_acc = evaluate(classifier, g, features, labels, test_mask)
print('Valid acc. %.2f Test acc. %.2f' % (valid_acc, test_acc))

if args.visualize:
    visualize_logits(g, all_logits)