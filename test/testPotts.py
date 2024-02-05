import polytensor
from polytensor.polynomial import PottsModel, PottsModelOneHot
import torch
from torch.nn import functional as F
import random
import numpy as np


def testPotts():
    print("\nTest Potts")

    num_bits = 6 #random.randint(5, 30)

    coefficients = polytensor.generators.coeffPUBORandomSampler(
        num_bits, [num_bits, 5, 5, 5], lambda: torch.rand(1)
    )

    p = PottsModel(coefficients)

    x = torch.tensor([[1, 0, 1, 2, 0, 1]])

    for item in coefficients.keys():
        print()
        #print(item, x[:, item], coefficients[item])
    #print("Coeff:\n", coefficients.keys())
    #print("Vals:\n", [x[:, key] for key in coefficients.keys()])
    print("Result", p(x))



    print("\nTest One Hot")

    num_bits = 6 #random.randint(5, 30)

    coefficients = polytensor.generators.coeffPUBORandomSampler(
        num_bits, [num_bits, 5, 5, 5], lambda: torch.rand(1)
    )

    p2 = PottsModelOneHot(coefficients)

    x = F.one_hot(torch.arange(0, num_bits), num_bits)

    for item in coefficients.keys():
        print()
        #print(item, x[:, item], coefficients[item])
    #print("Coeff:\n", coefficients.keys())
    #print("Vals:\n", [x[:, key] for key in coefficients.keys()])
    print("One-hot result", p2(x))


    '''
    coefficients = polytensor.generators.coeffPUBORandomSampler(
        num_bits, [num_bits, 1, 1, 1], lambda: torch.rand(1)
    )
    print(coefficients)

    num_classes = 5
    size =  (4, 4, 4)
    batch_size = 10

    indices = torch.randint(0, num_classes, size=size)
    one_hot_encoded = torch.nn.functional.one_hot(indices, num_classes=num_classes)
    one_hot_batch = one_hot_encoded.unsqueeze(0).expand(100, -1, -1, -1, -1)
    print(one_hot_batch)
    print(one_hot_batch.shape)
    '''

testPotts()
