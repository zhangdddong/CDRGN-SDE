# CDRGN-SDE: Cross-Dimensional Recurrent Graph Network with Neural Stochastic Differential Equation for Temporal Knowledge Graph Embedding

This repository provides the official PyTorch implementation of the research paper **CDRGN-SDE: Cross-Dimensional Recurrent Graph Network with Neural Stochastic Differential Equation for Temporal Knowledge Graph Embedding**. It will update soon. Thanks for your attention.

### Requirements

- python >= 3.6
- torch >= 1.8.0
- torchvision >= 0.9.2
- torch-scatter >= 2.0.6
- dgl-cu111 >= 0.6.1
- tqdm
- pandas
- rdflib

### How to Run

Step 1: pre-process the dataset

```python
python data/dataset/ent2word.py
python get_neighbor.py
```

Step 2: training the model

```python
python src/main.py
```
