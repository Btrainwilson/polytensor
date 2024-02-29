import polytensor
from polytensor.polynomial import PottsModel, PottsModelOneHot
import torch
from torch.nn import functional as F
import random
import numpy as np

import matplotlib

matplotlib.use("svg")

tests = [
    {
        "c": {
            (0, 1): -1,
        },
        "x": torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]]),
        "y": torch.Tensor([0.0, 0.0, -1.0, -1.0]),
    },
    {
        "c": {
            (0, 1): -1,
        },
        "x": torch.tensor([[2, 1], [0, 5], [4, 4], [2, 2]]),
        "y": torch.Tensor([0.0, 0.0, -1.0, -1.0]),
    },
    {
        "c": {
            (0, 1): -1,
        },
        "x": torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]]),
        "y": torch.Tensor([0.0, 0.0, -1.0, -1.0]),
    },
    {
        "c": {
            (0, 1): -1,
            (1, 2): -1,
        },
        "x": torch.tensor([[1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]]),
        "y": torch.Tensor([-2.0, 0.0, -1.0, -1.0]),
    },
    {
        "c": {
            (0, 1): -1,
            (1, 2): -1,
        },
        "x": torch.tensor([[1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]]),
        "y": torch.Tensor([-2.0, 0.0, -1.0, -1.0]),
    },
]


def testPotts():
    for test in tests:
        p = PottsModel(test["c"])
        x = test["x"]
        y = test["y"]

        assert np.allclose(
            p(x).detach().cpu().numpy(),
            y.detach().numpy(),
        )


def testPottsOneHot():
    for test in tests:
        p = PottsModelOneHot(test["c"])
        x = test["x"]
        y = test["y"]

        x = F.one_hot(x)

        assert np.allclose(
            p(x).detach().cpu().numpy(),
            y.detach().numpy(),
        )