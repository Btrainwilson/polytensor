import polytensor
import torch
import random
import numpy as np


def testPolynomial():
    print("\nTest 1: Generate from Dictionary of Coefficients")

    print("\nTest 2: Sparse vs Dense Polynomial")
    for _ in range(10):
        num_bits = random.randint(5, 30)

        coefficients = polytensor.generators.coeffPUBORandomSampler(
            num_bits, [num_bits, 5, 5, 5], lambda: torch.rand(1)
        )

        p = polytensor.SparsePolynomial(coefficients)

        x = torch.bernoulli(torch.ones(num_bits) * 0.5)

        q = polytensor.DensePolynomial(
            coefficients=polytensor.generators.denseFromSparse(
                coefficients, num_bits=num_bits
            )
        )

        assert np.allclose(p(x).detach().cpu().numpy(), q(x).detach().cpu().numpy())

    terms = {
        tuple([0]): 1.0,  # 1.0 * x_0
        tuple([1]): 2.0,  # 2.0 * x_1
        (0, 1): 3.0,  # 3.0 * x_0 * x_1
        (1, 1): 5.0,  # 5.0 * x_1^2
    }

    poly = polytensor.SparsePolynomial(terms)

    x = torch.Tensor([1.0, 2.0])

    # Evaluate the polynomial at x
    y_p = poly(x)

    # Which is equivalent to
    y_s = 0.0

    for term, v in terms.items():
        y_s += v * torch.prod(x[..., term])

    assert np.allclose(y_p.detach().cpu().numpy(), y_s.detach().cpu().numpy())
