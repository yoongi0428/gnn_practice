# Graph Neural Networks Practice
Graph neural networks (GNN) are drawing more and more attention and achieving remarkable improvement in various domains. Inspired by impressive [recent paper](https://arxiv.org/abs/2003.00982) benchmarking different GNNs, this repository aims to catch up many existing GNN variants without complicated codes and experiments.

Codes are implemented with [PyTorch](pytorch.org) and [DGL](https://docs.dgl.ai/index.html). For DGL, you can find kind [tutorial](https://docs.dgl.ai/en/0.4.x/tutorials/basics/1_first.html) on the official website. Codes refers to many part of codes from DGL tutorial (especially GCN, GAT).

## Cora Citation Network dataset
[Cora Citation Network dataset](http://eliassi.org/papers/ai-mag-tr08.pdf) is a transductive graph dataset with scientific papers as nodes and citation relationship as edges. It contains  2.7K nodes and 5.4K edges. Given a citation graph, the task is to classify each node to category it belongs to. Total number of category is 7.

## GNN Models
The following models are currently implemented. More to be added. 
- Graph Convolution Networks (Kipf & Welling, [*Semi-Supervised Classification with Graph Convolutional Networks*](https://arxiv.org/abs/1609.02907))
- Graph Attention Networks (Veličković et al., [*Graph Attention Networks*](https://arxiv.org/abs/1710.10903))

## Usage
`run.py` is all you need to run. All arguments for experiment settings and GNNs are set with `argparse` package and they can be found in [`utils/config.py`](utils/config.py).

For example, if you want to run **2-layer GCN Classifier** with `hidden_dim=50`, run the script below:
```
python run.py --gnn GCN --hidden_dim 50 --num_layers 2
```
Other parameters, such as learning rate or # of epochs, are set as default values.

**HAVE FUN !**

## Performance
Table below shows experimental results from the original paper and the current repo on Cora dataset (without exhaustive parameter search). 

|      Model       | Accuracy |
| ---------------- |:--------:|
|  GCN (paper)     |  81.5    |
|  GCN (repo)      |  80.6    |
|  GAT (paper)     |  83.0    |
|  GAT (repo)      |  82.0    |