"""
evaluation stage:
 - plot of the loss
 - test model on unseen training data
"""
import torch
from my_helpers.in_out import read_pickle
import matplotlib.pyplot as plt
import seaborn as sns
from modelling import conts_test, cats_test, y_test, TabularModel, criterion, params_model

train_progress = read_pickle('log_training_progress.pk')

plt.plot(train_progress['epoch'], train_progress['losses'])
plt.show()

model: TabularModel = TabularModel(**params_model)
model.load_state_dict(torch.load('first_model.pt'))

with torch.no_grad():
    y_pred = model.forward(conts_test, cats_test)
    loss = torch.sqrt(criterion(y_pred, y_test))
    print(f'RMSE: {loss}')

    residuals = y_pred - y_test

sns.distplot(residuals.numpy())
plt.show()
