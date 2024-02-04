from polytensor.polynomial import PottsModel, PottModelOneHot
import torch
from torch.nn import functional as F
import random
import numpy as np


def testPotts():
    print("\nTest 1")

    num_bits = 6 #random.randint(5, 30)

    coefficients = polytensor.generators.coeffPUBORandomSampler(
        num_bits, [num_bits, 5, 5, 5], lambda: torch.rand(1)
    )

    p = polytensor.PottsModel(coefficients)
    p2 = PottsModelOneHot(coefficients)

    x = torch.tensor([[1, 0, 1, 2, 0, 1]])

    print(p(x))
    print(p2(x))


testPotts()
