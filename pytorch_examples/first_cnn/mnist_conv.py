import pickle
from collections import defaultdict
from typing import List, DefaultDict
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import my_torch_helpers as mth

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='.', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='.', train=False, download=True, transform=transform)

BATCH_SIZE: int = 10
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

print(type(train_data))
print(train_data)


class ConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1)  # 6 filters > pooling > conv2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1)  # 16 filters, 3by3
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        layers: List = []
        layers.append(self.conv1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 16)  # a batch of one image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


model = ConvNetwork()
print(model)

num_params: int = 0
for param in model.parameters():  # less parameters than the ann equivalent
    print(param.numel())
    num_params += param.numel()
print(f'total params: {num_params}')

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
epochs: int = 2
monitoring: DefaultDict = defaultdict(list)

start_time = time.time()

for i in range(epochs):
    train_crt: int = 0
    test_crt: int = 0

    # train
    for b, (x_train, y_train) in enumerate(train_loader):
        b += 1
        pred = model.forward(x_train)
        loss = criterion(pred, y_train)

        batch_crt: int = (pred == y_train).sum()
        train_crt += batch_crt

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        mth.show_train_stats(i, b, loss.item())

    monitoring['train_correct'] = train_crt
    monitoring['train_loss'] = loss

    # test
    with torch.no_grad():
        for b, (x_test, y_test) in enumerate(test_loader):
            pred_test = model.forward(x_test)
            loss_test = criterion(pred_test, y_test)
            test_crt += (pred_test == y_test).sum()

        monitoring['test_correct'] = test_crt
        monitoring['test_loss'] = loss_test

with open('monitoring_minst.pk', 'wb') as f:
    pickle.dump(monitoring, f)
model.state_dict('cnn_minst.pt')

# confusion matrix is a good way to see where the model performs well
