"""
simple tabular pytorch model

difference between regression and classification
 - number of output layer nodes (regression=1, classification=k)
 - the loss function (MSE vs Cross Entropy Loss)
 -
"""

import time
from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, DefaultDict, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from cleaning import cols_categories, cols_number
from definitions import PROGRESS_FILE

from my_helpers.in_out import write_pickle

df = pd.read_pickle('taxi_cleaned')


# converting to tensors and category embedding #########################################################################

def embedding_vector_length(size: int) -> int:
    return min(50, (size + 1) // 2)


def show_sizes(*arrays) -> None:
    for arr in arrays:
        print(arr.shape)


class TabularModel(nn.Module):
    def __init__(self, embedding_sizes: List[Tuple[int, int]], num_vars_cont: int, num_out_nodes: int,
                 layer_sizes: List[int], prob_dropout: float):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(num_cats, vec_len) for num_cats, vec_len in embedding_sizes])
        self.embedding_dropout = nn.Dropout(prob_dropout)
        self.cont_batch_norm = nn.BatchNorm1d(num_vars_cont)

        layer_list: List = []
        num_embeddings: int = sum([num_emb for _, num_emb in embedding_sizes])
        num_ins: int = num_embeddings + num_vars_cont

        for num in layer_sizes:
            layer_list.append(nn.Linear(num_ins, num))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.BatchNorm1d(num))
            layer_list.append(nn.Dropout(prob_dropout))
            num_ins = num

        layer_list.append(nn.Linear(layer_sizes[-1], num_out_nodes))
        self.layers = nn.Sequential(*layer_list)

    def forward(self, conts, cats):
        embds = []
        for i, emb in enumerate(self.embeddings):
            embds.append(emb(cats[:, i]))

        embds_all = torch.cat(embds, 1)
        embds_all = self.embedding_dropout(embds_all)

        conts_norm = self.cont_batch_norm(conts)
        x = torch.cat([embds_all, conts_norm], 1)
        return self.layers(x)


cats = np.stack([df[col].cat.codes.values for col in cols_categories], axis=1)
conts = np.stack([df[col].values for col in cols_number], axis=1)

cats = torch.tensor(cats, dtype=torch.int64)
conts = torch.tensor(conts, dtype=torch.float)
y = torch.tensor(df['fare_amount'].values, dtype=torch.float).reshape(-1, 1)

cat_sizes = [len(df[col].cat.categories) for col in cols_categories]
emb_sizes = [(size, embedding_vector_length(size)) for size in cat_sizes]

params_model: Dict[str, Any] = {
    'embedding_sizes': emb_sizes,
    'num_vars_cont': conts.shape[1],
    'num_out_nodes': 1,
    'layer_sizes': [200, 100],
    'prob_dropout': 0.4,
}
model: TabularModel = TabularModel(**params_model)

criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size: int = conts.shape[0] // 2
test_size: int = int(batch_size * 0.2)

conts_train = conts[:batch_size - test_size]
cats_train = cats[:batch_size - test_size]
y_train = y[:batch_size - test_size]

conts_test = conts[batch_size - test_size: batch_size]
cats_test = cats[batch_size - test_size: batch_size]
y_test = y[batch_size - test_size: batch_size]

epochs: int = 300
data: DefaultDict[str, list] = defaultdict(list)
start_time = time.time()

if __name__ == '__main__':
    for i in range(epochs):
        i += 1
        y_pred = model.forward(conts_train, cats_train)
        loss = torch.sqrt(criterion(y_train, y_pred))

        time_taken = timedelta(seconds=round(time.time() - start_time))
        if i % 5 == 0:
            progress_msg: str = f'epoch: {i:2} of {epochs}, time taken: {str(time_taken)}, loss: {loss}'
            print(progress_msg)

        data['epoch'].append(i)
        data['losses'].append(loss)
        data['time_taken'].append(time_taken)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    write_pickle(data, str(PROGRESS_FILE))
    torch.save(model.state_dict(), 'first_model.pt')