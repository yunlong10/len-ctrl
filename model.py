import torch
import json
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        key = list(self.data.keys())[index]
        target = self.data[key]['target']
        seq = self.data[key]['seq']
        label = self.data[key]['label']
        return target, seq, label

    def __len__(self):
        return len(self.data)



class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[-1])
        return x

if '__name__' == '__main__':


    with open('data.json', 'r') as f:
        data = json.load(f)

    dataset = MyDataset(data)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    model = MyModel()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    num_epochs = 10

    # 训练模型
    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
            target, seq, label = data
            optimizer.zero_grad()
            output = model(target, seq)
            # 计算损失，注意要使用 label 与输出结果之和来计算损失
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        
        #
        # 在每个epoch结束后在验证集上验证模型
        with torch.no_grad():
            correct = 0
            total = 0
            for i, data in enumerate(val_dataloader):
                target, seq, label = data
                output = model(target, seq)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            val_acc = 100 * correct / total
            print('Validation accuracy: {}'.format(val_acc))


    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(test_dataloader):
            target, seq, label = data
            output = model(target, seq)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        
        test_acc = 100 * correct / total
        print('Test accuracy: {}'.format(test_acc))
