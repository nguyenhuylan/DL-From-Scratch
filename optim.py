"""
Use optimizer to adjust the parameters of our network based on the gradients
computed during backpropagation
"""

from nn import NeuralNet


class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.prams_and_grads():
            param -= self.lr * grad
