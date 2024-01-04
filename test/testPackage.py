import polytensor
import pytest
import numpy as np
import torch


def testPolynomial():
    print("\nTest 1: Generate from Coefficients")

    coefficients = {
        (0, 1): 2,
        (0, 0): -1,
        (1, 1): 4,
    }
    print("coefficients:", coefficients)
    p = polytensor.SparsePolynomial(coefficients)
    print("p:", p)
    x = torch.tensor([0, 1], dtype=torch.float)
    print("x:", x)

    print("\nTest 2: Generate from Tensors")

    tensors = polytensor.generators.denseFromSparse(
        coefficients, lambda n, deg: torch.rand(1)
    )

    q = polytensor.DensePolynomial(coefficients=tensors)
    print("p(x):", p(x))
    print("q(x):", q(x))
