import torch
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def encode_onehot(labels):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)

    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    # reshape labels to a col based np array
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))

    return onehot_encoded


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # Convert it to a sparse matrix in coordinate format
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def load_data(path="../data/cora/", dataset="cora"):
    print('Loading {} dataset...'.format(dataset))

    # load dataset from file as a numpy array
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))

    # choose all rows and second to penultimate cols
    # which .content format is <paper_id> <word_attributes>+ <class_label>
    # Using CSR_ Matrix compresses and stores feature data,
    # effectively storing non-zero elements and their positional information of sparse matrices
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))  # Convert to dense tensor format in Torch

    # extract labels and use one-hot to encode it
    labels = encode_onehot(idx_features_labels[:, -1])
    # np.where used for returning index of non-zero element(row index and col index)
    # row index converted to LongTensor which is th expected type of classification in pytorch
    labels = torch.LongTensor(np.where(labels)[1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}  # key: ID of cited paper, value: index in idx (node number)

    # <ID of cited paper> <ID of citing paper>
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # >>> a = np.array([[1,2], [3,4]])
    # >>> a.flatten()
    # array([1, 2, 3, 4])
    # after that, it makes this format of np array
    # <id5, id2> -> map.get(id5) = node7, map.get(id2) = node12
    # [[7, 12], ....]
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    # np.ones create a np array which length is all elements are 1, this np array represents adj
    #  >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
    #  >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
    #  >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
    #     array([[3, 0, 1, 0],
    #            [0, 2, 0, 0],
    #            [0, 0, 0, 0],
    #            [0, 0, 0, 1]])
    # coo[ row[0], col[0] ] = data[0] -> coo[0,0] += 1
    # (edges[: 0], edges[:, 1]):
    # row = np.array([..., 7, ...])
    # col = np.array([..., 12, ...])
    # coo[7, 12] += 1, but finally coo[7, 12] = 1
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # sp.eye(adj.shape[0]): The element on the diagonal is 1
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # change to sparse format float tensor
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = range(140)
    idx_validate = range(200, 500)
    idx_test = range(500, 1500)

    idx_train = torch.LongTensor(idx_train)
    idx_validate = torch.LongTensor(idx_validate)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_validate, idx_test
