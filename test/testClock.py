import polytensor
from polytensor.polynomial import ClockModelOneHot
import torch
from torch.nn import functional as F
import random
import numpy as np

tests = [
    {
        "c": {
            (0, 1): 1,
        },
        "x": torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]]),
        "y": torch.Tensor([1.0, 1.0, -1.0, -1.0]),
    },
    {
        "c": {
            (0, 1): -1,
        },
        "x": torch.tensor([[2, 1], [0, 5], [4, 4], [2, 2]]),
        "y": torch.Tensor([-1.0, 0.125, -0.5, -0.5]),
    },
    {
        "c": {
            (0, 1): -1,
        },
        "x": torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]]),
        "y": torch.Tensor([-1.0, -1.0, 1.0, 1.0]),
    },
    {
        "c": {
            (0, 1): -1,
            (1, 2): -1,
        },
        "x": torch.tensor([[1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]]),
        "y": torch.Tensor([2.0, -2.0, 0.0, 0.0]),
    },
    {
        "c": {
            (0, 1): -1,
            (1, 2): -1,
        },
        "x": torch.tensor([[2, 1, 3], [0, 5, 2], [4, 4, 4], [2, 2, 0]]),
        "y": torch.Tensor([-1.875, -0.875, -1.0, -1.375]),
    },
]

def testClockOneHot():
    for test in tests:
        p = ClockModelOneHot(test["c"])
        x = test["x"]
        y = test["y"]

        x = F.one_hot(x)

        assert np.allclose(
            p(x.float()).view(-1).detach().cpu().numpy(),
            y.detach().numpy(),
        )

testClockOneHot()