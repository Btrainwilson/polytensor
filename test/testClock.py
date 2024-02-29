import polytensor
from polytensor.polynomial import ClockModelOneHot
import torch
from torch.nn import functional as F
import random
import numpy as np


def testClockOneHot():
    print("\nTest Clock One Hot")

    num_bits = 6 #random.randint(5, 30)

    coefficients = polytensor.generators.coeffPUBORandomSampler(
        num_bits, [num_bits, 5, 5, 5], lambda: torch.rand(1)
    )

    p = ClockModelOneHot(coefficients)

    x = torch.tensor([[1, 0, 3, 2, 0, 1]])
    x = F.one_hot(x).float()
    
    print("Result", p(x))


def testClockOneHotMultiDimension():
    print("\nTest Clock One Hot Multi Dimension")

    num_bits = 6 #random.randint(5, 30)

    # size = (4, 2, 2, 6)
    size = (4, 2, 6)

    coefficients = polytensor.generators.coeffPUBORandomSampler(
        num_bits, [num_bits, 5, 5, 5], lambda: torch.rand(1)
    )
    print(coefficients)

    p = ClockModelOneHot(coefficients)

    x = torch.randint(0, 3, size=size)
    print("x: ", x.shape)

    x = F.one_hot(x).float()

    print("x one hot: ", x.shape)

    result = p(x)
    print("Result", result.shape)
    print(result)


testClockOneHot()
testClockOneHotMultiDimension()
