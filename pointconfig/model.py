import torch
from torch import nn
import numpy as np

import pointconfig.lightweight_score as lws
import pointconfig.make_subset as ms

PRIME = lws.PRIME
DIMENSION = lws.DIMENSION
WORD_LENGTH = lws.WORD_LENGTH
INTPUT_LENGTH = ms.INPUT_LENGTH

FIRST_LAYER = 128
SECOND_LAYER = 64
THIRD_LAYER = 4

LEARNING_RATE = 1e-3


def model_info():
    model = nn.Sequential(
        nn.Linear(INTPUT_LENGTH, FIRST_LAYER),
        nn.ReLU(),
        nn.Linear(FIRST_LAYER, SECOND_LAYER),
        nn.ReLU(),
        nn.Linear(SECOND_LAYER, THIRD_LAYER),
        nn.ReLU(),
        nn.Linear(THIRD_LAYER, 1),
        nn.Sigmoid(),
    )

    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return model, loss_function, optimizer


def train_model(
    data, model, loss_function, optimizer, batch_size=1024, shuffle=True
):
    data_size = data.shape[0]
    if shuffle:
        indices = torch.randperm(data_size)
        data = data[indices, :]
    model.train()
    total_loss = 0
    for start in range(0, data_size, batch_size):
        end = start + batch_size
        data_input = data[start:end, :-1].float()
        data_true = data[start:end, -1].float().unsqueeze(1)

        data_pred = model(data_input)
        loss = loss_function(data_pred, data_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data_input.size(0)

    return total_loss / data_size


def model_forward(data_input, model):
    with torch.no_grad():
        return model(data_input).flatten()
