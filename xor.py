"""XOR"""

import torch
from torch import Tensor

from train import train
from nn import NeuralNet
from layers import Linear, Tanh

inputs = Tensor([
    [0, 0],
    [1, 0],
    [0, 1],
    [0, 0]
])

targets = Tensor([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)