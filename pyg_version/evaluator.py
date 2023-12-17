import torch
import torch.nn.functional as F
from data_loader import pyg_loader
from model import GCN
import matplotlib.pyplot as plt


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

    train_accuracies = []  # 用于记录每个训练周期结束后的训练精度

    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        out = model(features, edges)
        loss = F.nll_loss(out[idx_train], labels[idx_train])  # 损失函数
        loss.backward()
        optimizer.step()

        # 在训练过程中计算训练精度并记录
        _, train_pred = out[idx_train].max(dim=1)
        correct_train = train_pred.eq(labels[idx_train]).sum()
        train_acc = int(correct_train) / int(len(idx_train))
        train_accuracies.append(train_acc)

        print(f"epoch:{epoch + 1}, loss:{loss.item()}, train_acc:{train_acc}")

    # 在训练结束后绘制训练精度图
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.legend()
    plt.show()

    model.eval()
    _, pred = model(features, edges).max(dim=1)
    correct = pred[idx_test].eq(labels[idx_test]).sum()  # 计算预测与标签相等个数
    acc = int(correct) / int(len(idx_test))  # 计算正确率
    print(acc)

