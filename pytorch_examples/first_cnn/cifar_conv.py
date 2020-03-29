"""cifar-10 dataset of 32x32 images of 10 different objects in color"""
import pickle
from collections import defaultdict
from typing import DefaultDict
import my_torch_helpers as mth
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

BATCH_SIZE: int = 10

transform = transforms.ToTensor()
train_data = datasets.CIFAR10(root='.', train=True, transform=transform, download=True)
test_data = datasets.CIFAR10(root='.', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

for i, (x_train, y_train) in enumerate(train_data):
    break

print(x_train)
print(x_train.shape)  # 3 color channels , 32 pix, 32 pix

for images, labels in train_loader:
    break


# im = make_grid(images, nrow=5)
# plt.figure(figsize=(10, 4))
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()

class ConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(6 * 6 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)
        x = x.view(-1, 6 * 6 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


model = ConvNetwork()
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters())
monitoring: DefaultDict = defaultdict(list)

epochs: int = 8

for i in range(epochs):
    train_crt: int = 0
    test_crt: int = 0

    # training batch loops
    for b, (x_train, y_train) in enumerate(train_loader):
        b += 1
        pred = model.forward(x_train)
        loss = criterion(pred, y_train)

        predicted = torch.max(pred.data, dim=1)[1]
        train_crt += (predicted == y_train).sum()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        accurancy = train_crt.item() / b * BATCH_SIZE
        mth.show_train_stats(i, b, loss.item(), accuracy=accurancy)

    monitoring['epoch'] = i
    monitoring['batch'] = b
    monitoring['train_loss'] = loss
    monitoring['train_crt'] = train_crt

    # testing batch loops
    with torch.no_grad():
        for b, (x_test, y_test) in enumerate(test_loader):
            pred_test = model.forward(x_test)
            less_test = criterion(pred_test, y_test)

            predicted = torch.max(pred_test, dim=1)[1]
            test_crt += (predicted == y_test).sum()

        monitoring['test_loss'] = loss
        monitoring['test_crt'] = test_crt

with open('cifar_mont.pk', 'wb') as f:
    pickle.dump(monitoring, f)

torch.save(model.state_dict(), 'cifar.pt')
