import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from dgl.data import citation_graph as citegrh
from dgl import DGLGraph

from utils.utils import set_random_seed, visualize_logits
from utils.config import load_arg_parser
from utils.trainer import load_cora_data, build_classifier, evaluate

args = load_arg_parser()

set_random_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists('best_model'):
    os.mkdir('best_model')

# Load cora dataset from DGL library
g, features, labels, train_mask, valid_mask, test_mask = load_cora_data(device)

# Load GNN Classifier
gnn = args.gnn
num_nodes, input_dim = features.shape
hidden_dim = args.hidden_dim
num_labels = int(labels.max() + 1)  # assume dataset covers all labels

classifier = build_classifier(gnn, input_dim, hidden_dim, num_labels, args).to(device)

# Experiment Settings
num_epochs = args.num_epochs
lr = args.lr
early_stop = args.early_stop
patience = 0
best_acc = -1
best_epoch = -1

optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

# Train Classifier
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

# Train Classifier
classifier.load_state_dict(torch.load(os.path.join('best_model', '%s_best.pt' % gnn)))
valid_acc = evaluate(classifier, g, features, labels, valid_mask)
test_acc = evaluate(classifier, g, features, labels, test_mask)
print('Valid acc. %.3f Test acc. %.3f' % (valid_acc, test_acc))

# Visualize classification through all training epochs
if args.visualize:
    visualize_logits(g, all_logits)