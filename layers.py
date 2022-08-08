"""
Neural network will be made up of layers.

"""
from typing import List, Dict, Callable
import torch
from torch import Tensor

class Layer:
    def __init__(self) -> None:
        self.params: Dict = {}
        self.grads: Dict = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the outputs corresponding to these inputs
        :param inputs:
        :return:
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate this gradient though the layer

        :param grad:
        :return:
        """
        raise NotImplementedError

class Linear(Layer):
    """
    Compute output = input @ w + b
    """
    def __init__(self, input_size: int,
                 output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self.params["w"] = torch.randn(input_size, output_size)
        self.params["b"] = torch.rand(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        output = inputs @ w + b
        :param inputs:
        :return:
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        If y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)

        :param grad:
        :return:
        """
        self.grads["b"] = torch.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    An activation layer just applies a function elementwise to its inputs
    """
    def __init__(self, f: F, f_prime:F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        If y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)

        :param grad:
        :return:
        """
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return torch.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2

class Tanh(Activation):
    def __init__(self):
        super().__init__()