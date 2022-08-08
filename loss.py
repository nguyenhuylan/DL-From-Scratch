from typing import Dict, List

import torch
from torch import Tensor

class Loss:
    def loss(self, predicted: Tensor, gt: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, gt: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """
    MSE is mean squared error
    """
    def loss(self, predicted: Tensor, gt: Tensor) -> Tensor:
        return torch.sum((predicted - gt) ** 2)

    def grad(self, predicted: Tensor, gt: Tensor) -> Tensor:
        return 2 * (predicted - gt)
