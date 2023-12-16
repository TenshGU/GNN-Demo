import math
import torch
from torch.nn import Module, Parameter


class GraphConv(Module):
    def __init__(self, input_features, output_features, bias=True):
        super(GraphConv, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = Parameter(torch.FloatTensor(input_features, output_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # X * W
        support = torch.mm(x, self.weight)
        # D^-1/2 * A * D^-1/2 * X * W
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_features) + ' -> ' \
            + str(self.output_features) + ')'

