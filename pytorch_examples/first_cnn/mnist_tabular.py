import time
from collections import defaultdict
from datetime import timedelta
from typing import List, Type, DefaultDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S',
    level=logging.INFO,
    # filename='logs.txt'
)

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='.', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='.', train=False, download=True, transform=transform)

image, label = train_data[0]

plt.imshow(image.reshape(28, 28), cmap='gist_yarg')

BATCH_SIZE_TRAIN: int = 100
BATCH_SIZE_TEST: int = 500
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE_TEST, shuffle=False)


class MultilayerModel(nn.Module):
    def __init__(self, num_ins: int, layer_sizes: List[int], num_outs: int, prob_dropout: float):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.num_outs = num_outs
        self.prob_dropout = prob_dropout

        layers: List = []
        for size in layer_sizes:
            layers.append(nn.Linear(num_ins, size))
            layers.append(nn.ReLU(inplace=True))
            num_ins = size

        layers.append(nn.Linear(size, num_outs))

        self.layers = nn.Sequential(*layers)

    def forward(self, variables: torch.Tensor):
        X = self.layers(variables)
        return F.log_softmax(X, dim=1)


def show_model(model: Type[nn.Module]) -> None:
    """print model and the number of parameters"""
    print(str(model) + '\n')
    total_params = sum([param.numel() for param in model.parameters()])
    print(f'total params: {total_params:,}')
    for param in model.parameters():
        print('\t' + str(param.numel()))


params = {
    "num_ins": 784,
    "layer_sizes": [120, 84],
    "num_outs": 10,
    "prob_dropout": 0.4,
}
model: MultilayerModel = MultilayerModel(**params)
show_model(model)

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()
epochs: int = 10
monitoring: DefaultDict = defaultdict(list)


def time_diff(start: float) -> timedelta:
    return timedelta(seconds=round(time.time() - start))


# there are multiple batches per epoch, an epoch is an entire run forward and backward through the model within each
# run we feed the images in via batches of the epoch

for i in range(epochs):

    train_correct: float = 0
    test_correct = 0
    for b, (x_train, y_train) in enumerate(train_loader):
        b += 1
        y_pred = model.forward(x_train.view(BATCH_SIZE_TRAIN, -1))
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (y_train == predicted).sum().item()
        train_correct += batch_corr

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if b % 200 == 0:
            acc: float = (train_correct * 100) / (b * 100)
            print(f'epoch: {i}, batch: {b}, time_taken: {time_diff(start_time)} loss: {loss:.3f}, accuracy: {acc:.3f}')

    monitoring['epoch '] = i
    monitoring['train_loss'] = loss
    monitoring['train_correct'] = train_correct

    with torch.no_grad():
        for b, (x_test, y_test) in enumerate(test_loader):
            y_pred_test = model.forward(x_test.view(BATCH_SIZE_TEST, -1))
            predicted_test = torch.max(y_pred_test.data, 1)[1]

            test_correct += (predicted_test == y_test).sum()

    loss = criterion(y_pred_test, y_test)
    monitoring['test_loss'] = loss
    monitoring['test_correct'] = test_correct

print(f'total time: {time_diff(start_time)}')

# saving the model

torch.save(model.state_dict(), 'ann.pt')
