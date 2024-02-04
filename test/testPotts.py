import polytensor
from polytensor.polynomial import PottsModel, PottsModelOneHot
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

    p = PottsModel(coefficients)
    p2 = PottsModelOneHot(coefficients)

    x = torch.tensor([[1, 0, 1, 2, 0, 1]])

    for item in coefficients.keys():
        print(item, x[:, item], coefficients[item])
    #print("Coeff:\n", coefficients.keys())
    #print("Vals:\n", [x[:, key] for key in coefficients.keys()])
    print("Result", p(x))
    print("One-hot result", p2(x))


testPotts()
