from torch import nn
from layer import GraphConv
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, num_feat, num_hidden, num_class, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConv(num_feat, num_hidden)
        self.gc2 = GraphConv(num_hidden, num_class)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
