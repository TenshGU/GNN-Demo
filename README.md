# Environment:

The dependencies involved have been written down in **requirements.txt**

run as follows:

`pip install -r requirements.txt`

# DataBase:

- cora
  - cora.cites
  - cora.content

# Implements:

## 0 data loader

The `.py files` in `data_loader package` are as follows:

- **loader.py** : loading cora dataset, extracting node features and edge information, and calculating D<sup>-1/2</sup> A D<sup>-1/2</sup>...

## 1 Basic Pytorch version：

The `.py files` in `basic_version package` are as follows:

- **layer.py** : Defined Graph Convolution Layer (GraphConv)
- **model.py** : Defined GCN Model which composed by GraphConv
- **evaluator.py** : Used to evaluate our model

## 2 PyG version

The `.py files` in `pyg_version package` are as follows:

- **model.py** : Defined GCN Model composed by `GCNConv` which provided by torch_geometric
- **evaluator.py** : Used to evaluate our model

# Result:
![image](https://github.com/TenshGU/GNN-Demo/master/results/pyg_model_res.png)
