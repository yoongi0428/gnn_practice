# Graph Neural Networks Practice
Following the increasing attention on and improvement of graph neural networks in various domains, this repository aims to catch up many existing GNN variants without complicated codes and experiments. 

Codes are implemented with [PyTorch](pytorch.org) and [DGL](https://docs.dgl.ai/index.html). For DGL, you can find kind [tutorial](https://docs.dgl.ai/en/0.4.x/tutorials/basics/1_first.html) on the official website.

## Cora Citation Network dataset
[Cora Citation Network dataset](http://eliassi.org/papers/ai-mag-tr08.pdf) is a graph with scientific papers as nodes and citation relationship as edges. It contains  2.7K nodes and 5.4K edges. Given a citation graph, the task is to classify each node (paper) to which category it belongs. Total number of category is 7.

## GNN Models
The following models are currently implemented. More to be added. 
- Graph Convolution Networks (Kipf & Welling, [*Semi-Supervised Classification with Graph Convolutional Networks*](https://arxiv.org/abs/1609.02907))
- Graph Attention Networks (Veličković et al., [*Graph Attention Networks*](https://arxiv.org/abs/1710.10903))

## Performance
Table below shows experimental results from the original paper and the current repo (without exhaustive parameter search).

|      Model       | Accuracy |
| ---------------- |:--------:|
|  GCN (paper)     |  81.5    |
|  GCN (repo)      |          |
|  GAT (paper)     |  83.0    |
|  GAT (repo)      |          |