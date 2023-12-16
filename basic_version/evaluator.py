import time
import torch
from data_loader import loader
from models import *


def accuracy(output, labels):
    predications = output.max(1)[1].type_as(labels)
    correct = predications.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(epoch, model, optimizer, features, adj, idx_train, labels, fastmode):
    t = time.time()
    model.train()
    optimizer.zero_grad()  # zero gradient
    output = model(features, adj)
    loss_train = torch.nn.functional.nll_loss(output[idx_train], labels[idx_train])  # loss function
    acc_train = accuracy(output[idx_train], labels[idx_train])  # calculate accuracy
    loss_train.backward()  # back propagation
    optimizer.step()  # update gradient

    if not fastmode:
        model.eval()
        output = model(features, adj)

    loss_val = torch.nn.functional.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(model, features, adj, idx_test, labels):
    model.eval()
    output = model(features, adj)  # features:(2708, 1433)   adj:(2708, 2708)
    loss_test = torch.nn.functional.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    hidden = 16  # define the number of hidden layers
    dropout = 0.5
    lr = 0.01
    weight_decay = 5e-4
    fastmode = 'store_true'
    epochs = 500

    adj, features, labels, idx_train, idx_val, idx_test = loader.load_data()
    model = GCN(num_feat=features.shape[1], num_hidden=hidden,
                num_class=labels.max().item() + 1, dropout=dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)
    features, adj, labels, idx_train, idx_val, idx_test = (
        item.to(device) for item in (features, adj, labels, idx_train, idx_val, idx_test)
    )
    for epoch in range(epochs):
        train(epoch, model, optimizer, features, adj, idx_train, labels, fastmode)
    test(model, features, adj, idx_test, labels)


