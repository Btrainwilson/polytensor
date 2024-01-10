import polytensor
import numpy as np
import torch


def testGrad():
    print("\nTest 1: Generate from Coefficients")

    orig_coefficients = polytensor.generators.randomConstant(
        5, num_terms=3, degree=2, sample_fn=lambda: torch.rand(1)
    )

    coefficients = orig_coefficients.copy()

    print("coefficients:", coefficients)

    true_coeff = coefficients.copy()
    for k, v in coefficients.items():
        coefficients[k] = torch.tensor(1, dtype=torch.float)

    p = polytensor.SparsePolynomial(true_coeff)
    q = polytensor.SparsePolynomial(coefficients)

    print(p.coeff_vector)

    for param in p.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(q.parameters(), lr=0.1)
    tloss = 0

    for i in range(10000):
        x = torch.bernoulli(torch.ones(5, device="cpu") * 0.5)
        y = p(x)
        z = q(x)
        loss = torch.sum((y - z) ** 2)
        optimizer.zero_grad()
        loss.backward()
        tloss += loss.item()
        optimizer.step()
        if i % 1000 == 0:
            tloss /= 1000
            print(tloss)
            tloss = 0

    print(p.coefficients)
    print(q.coefficients)
