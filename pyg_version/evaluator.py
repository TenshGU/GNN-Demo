import torch
import torch.nn.functional as F
from data_loader import pyg_loader
from model import GCN


if __name__ == "__main__":
    hidden = 16
    dropout = 0.5
    weight_decay = 5e-4
    lr = 0.01

    features, edges, labels = pyg_loader.load_data()
    idx_train = range(2000)  # 其中2000个点是训练数据
    idx_test = range(2000, 2700)  # 700个测试数据
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(features.shape[1], hidden, labels.max().item() + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        out = model(features, edges)
        loss = F.nll_loss(out[idx_train], labels[idx_train])  # 损失函数
        loss.backward()
        optimizer.step()
        print(f"epoch:{epoch + 1}, loss:{loss.item()}")

    model.eval()
    _, pred = model(features, edges).max(dim=1)
    correct = pred[idx_test].eq(labels[idx_test]).sum()  # 计算预测与标签相等个数
    acc = int(correct) / int(len(idx_test))  # 计算正确率
    print(acc)

