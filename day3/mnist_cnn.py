import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.out(x)
        y = torch.log_softmax(x, dim=-1)

        return y

if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_loss(label, pred):
        return criterion(pred, label)

    def train_step(x, t):
        model.train()
        preds = model(x)
        loss = compute_loss(t, preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, preds

    def test_step(x, t):
        model.eval()
        preds = model(x)
        loss = compute_loss(t, preds)

        return loss, preds

    # load data
    root = os.path.join(os.path.dirname('__file__'), '..', 'data',
    'mnist')
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = \
        torchvision.datasets.MNIST(root=root,
                                   download=True,
                                   train=True,
                                   transform=transform)
    mnist_test = \
        torchvision.datasets.MNIST(root=root,
                                   download=True,
                                   train=False,
                                   transform=transform)

    n_samples = len(mnist_train)
    train_size = int(len(mnist_train) * 0.9)
    val_size = n_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(mnist_train, [train_size, val_size])


    train_dataloader = DataLoader(train_dataset,
                                  batch_size=100,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                  batch_size=100,
                                  shuffle=True)
    test_dataloader = DataLoader(mnist_test,
                                batch_size=100,
                                shuffle=False)

    # Build model
    model = CNN().to(device)
    criterion = nn.NLLLoss()
    optimizer = optimizers.Adam(model.parameters())

    # Train model
    epochs = 10

    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        val_acc = 0

        for (x, t) in train_dataloader:
            x, t = x.to(device), t.to(device)
            loss, _ = train_step(x, t)
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        train_loss_list.append(train_loss)

        for (x, t) in val_dataloader:
            x, t = x.to(device), t.to(device)
            loss, preds = test_step(x, t)
            val_loss += loss.item()
            val_acc += \
                accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())


        val_loss /= len(val_dataloader)
        val_loss_list.append(val_loss)
        val_acc /= len(val_dataloader)
        print('Epoch: {} / {}, Valid Cost: {:3f}, Valid Acc: {:3f}'.format(
            epoch+1,
            epochs,
            val_loss,
            val_acc
        ))


    # plot
    x_axis = [i in range(epochs)]
    fig = plt.figure()
    ax.plot(x_axis, train_loss_list, label="train loss")
    ax.plot(x_axis, val_loss_list, label="valid loss")
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')

    ax.legend(loc='best')
    ax.set_title('Train, Valid loss')
    plt.savefig('figure.png')

    test_loss = 0
    test_acc = 0

    for (x, t) in test_dataloader:
        x, t = x.to(device), t.to(device)
        loss, preds = test_step(x, t)
        test_loss += loss.item()
        test_acc += \
            accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    print('Test Loss : {:3f}, Test Acc: {:3f}'.format(
        epoch+1,
        test_loss,
        test_acc
    ))
