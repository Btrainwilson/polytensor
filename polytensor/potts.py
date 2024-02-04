import torch
from .polynomial import SparsePolynomial
import numpy as np
from beartype import beartype
from beartype.typing import List, Union
from enum import Enum
from abc import ABC, abstractmethod
import logging


class PottsModel(SparsePolynomial):

    def __init__(
        self,
        coefficients: dict[List[int], Union[complex, float, int, torch.Tensor]],
        device: str = "cpu",
        dtype=torch.float,
        **kwargs,
    ):
        super().__init__(coefficients, device, dtype, **kwargs)

    def forward(self, x):
      
        r = False
        sum = 0.0

        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            r = True

        for key, v in self.coefficients.items():
            sum = sum - v * torch.eq(torch.max(x[:, key], 1).values, torch.min(x[:, key], 1).values)

        if r:
            return sum.squeeze()

        return sum
