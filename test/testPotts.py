<<<<<<< HEAD
from polytensor.polynomial import PottsModel, PottModelOneHot
=======
import polytensor
from polytensor.polynomial import PottsModel, PottsModelOneHot
>>>>>>> a13df790984e5818c21f4fdbffdd5a4a1b629c60
import torch
from torch.nn import functional as F
import random
import numpy as np


def testPotts():
<<<<<<< HEAD
    print("\nTest 1")
=======
    print("\nTest Potts")
>>>>>>> a13df790984e5818c21f4fdbffdd5a4a1b629c60

    num_bits = 6 #random.randint(5, 30)

    coefficients = polytensor.generators.coeffPUBORandomSampler(
        num_bits, [num_bits, 5, 5, 5], lambda: torch.rand(1)
    )

<<<<<<< HEAD
    p = polytensor.PottsModel(coefficients)
    p2 = PottsModelOneHot(coefficients)

    x = torch.tensor([[1, 0, 1, 2, 0, 1]])

    print(p(x))
    print(p2(x))


testPotts()
=======
    p = PottsModel(coefficients)

    x = torch.tensor([[1, 0, 1, 2, 0, 1]])

    for item in coefficients.keys():
        print(item, x[:, item], coefficients[item])
    #print("Coeff:\n", coefficients.keys())
    #print("Vals:\n", [x[:, key] for key in coefficients.keys()])
    print("Result", p(x))

def testPottsMultiDimension():
    print("\nTest Potts Multi Dimension")

    num_bits = 6 #random.randint(5, 30)
    dimensions = 1
    batch_size = 5
    size = (batch_size,) + (num_bits,) * dimensions
    size = (4, 2, 2, 6)

    coefficients = polytensor.generators.coeffPUBORandomSampler(
        num_bits, [num_bits, 5, 5, 5], lambda: torch.rand(1)
    )
    print(coefficients)

    p = PottsModel(coefficients)

    x = torch.randint(0, 3, size=size)
    print("x: ", x.shape)
    print(x)

    result = p(x)
    print("Result", result.shape)
    print(result)


def testPottsOneHot():

    print("\nTest One Hot")

    num_bits = 6 #random.randint(5, 30)

    coefficients = polytensor.generators.coeffPUBORandomSampler(
        num_bits, [num_bits, 5, 5, 5], lambda: torch.rand(1)
    )

    p1 = PottsModel(coefficients)
    p2 = PottsModelOneHot(coefficients)

    size = (6,)
    x = torch.randint(0, 3, size=size)
    og = x

    print("x", x.shape)
    print(x)

    x = F.one_hot(x)
    print("x", x.shape)
    print(x)

    result = p2(x)
    print("One-hot result", result.shape)
    print(result)

    print("Normal result", p1(og))

def testPottsOneHotMultiDimension():

    print("\nTest One Hot Multi Dimension")

    num_bits = 6 #random.randint(5, 30)

    coefficients = polytensor.generators.coeffPUBORandomSampler(
        num_bits, [num_bits, 5, 5, 5], lambda: torch.rand(1)
    )

    p1 = PottsModel(coefficients)
    p2 = PottsModelOneHot(coefficients)

    size = (6, 5, 6)
    x = torch.randint(0, 3, size=size)

    print("x", x.shape)
    print(x)

    og = x
    x = F.one_hot(x)
    print("x", x.shape)
    print(x)

    result = p2(x)
    print("One-hot result", result.shape)
    print(result)

    print("Normal result", p1(og))

def testB():
    print("\nTest B")

    num_bits = 5 #random.randint(5, 30)
    coefficients = polytensor.generators.coeffPUBORandomSampler(
        num_bits, [num_bits, 5, 5, 5], lambda: torch.rand(1)
    )
    p1 = PottsModel(coefficients)
    p2 = PottsModelOneHot(coefficients)

    size = (6, 5, 5, 5)
    x = torch.randint(0, 3, size=size)

    og = x
    x = F.one_hot(x)
    print("x", x.shape)
    print(x)

    result = p1(og)
    result2 = p2(x)

    print(result.shape, result2.shape)
    print(result)
    print(result2)


testPotts()
testPottsMultiDimension()
testPottsOneHot()
testPottsOneHotMultiDimension()
testB()
>>>>>>> a13df790984e5818c21f4fdbffdd5a4a1b629c60
