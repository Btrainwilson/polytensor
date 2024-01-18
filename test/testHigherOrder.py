import polytensor
import torch
from torch.nn import functional as F
import random
import numpy as np


def testCategorical():
    print("\nTest 1: One-Hot")

    num_bits = random.randint(5, 30)

    coefficients = polytensor.generators.coeffPUBORandomSampler(
        num_bits, [num_bits, 5, 5, 5], lambda: torch.rand(1)
    )

    p = polytensor.SparsePolynomial(coefficients)

    x = torch.bernoulli(torch.ones(num_bits) * 0.5)

    x = F.one_hot(torch.arange(0, num_bits), num_bits)

    print(p(x))


testCategorical()
