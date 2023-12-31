import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feature, hidden)
        self.conv2 = GCNConv(hidden, classes)

    def forward(self, features, edges):
        features = self.conv1(features, edges)
        features = F.relu(features)
        features = F.dropout(features, training=self.training)
        features = self.conv2(features, edges)
        return F.log_softmax(features, dim=1)
