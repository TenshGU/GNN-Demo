import torch
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize as normalizer


def encode_onehot(labels):  # 把标签转换成onehot
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),  # 读取节点特征和标签
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 读取节点特征
    dict = {int(element): i for i, element in enumerate(idx_features_labels[:, 0:1].reshape(-1))}  # 建立字典
    labels = encode_onehot(idx_features_labels[:, -1])  # 标签用onehot方式表示
    e = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)  # 读取边信息
    edges = []
    for i, x in enumerate(e):
        edges.append([dict[e[i][0]], dict[e[i][1]]])  # 若A->B有变 则B->A 也有边
        edges.append([dict[e[i][1]], dict[e[i][0]]])  # 给的数据是没有从0开始需要转换
    features = normalizer(features)  # 特征值归一化
    features = torch.tensor(np.array(features.todense()), dtype=torch.float32)
    labels = torch.LongTensor(np.where(labels)[1])
    edges = torch.tensor(edges, dtype=torch.int64).T
    return features, edges, labels
