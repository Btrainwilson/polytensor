import polytensor
import matplotlib.pyplot as plt
import numpy as np
import time
import torch

import matplotlib

matplotlib.use("svg")


def testGraphDenseVsSparse():
    print("\nTest Graph Dense Vs Sparse")
    b = 10
    num_bits = 1000
    num_terms = 1000

    # Lists to store degree numbers and corresponding times
    degrees = []
    sparseTimes = []
    denseTimes = []

    for degreeNum in range(2, 5):

        degrees.append(degreeNum)
        coefficients = polytensor.generators.coeffRandomSampler(
            num_bits, num_terms, degreeNum, lambda: torch.rand(1, device="cuda")
        )

        x = torch.bernoulli(torch.ones(b, num_bits, device="cuda") * 0.5)
        p = polytensor.SparsePolynomial(coefficients, device="cuda")
        q = polytensor.DensePolynomial(
            coefficients=polytensor.generators.denseFromSparse(
                coefficients, num_bits=num_bits, device="cuda"
            ),
            device="cuda",
        )
        with torch.no_grad():
            p(x)
            startSparse = time.time()
            p(x)
            stopSparse = time.time()

            sparseTimes.append(stopSparse - startSparse)

            q(x)
            startDense = time.time()
            q(x)
            stopDense = time.time()

        denseTimes.append(stopDense - startDense)

        # assert np.allclose(p(x).detach().cpu().numpy(), q(x).detach().cpu().numpy())
        # print(coefficients)

    # Plotting the graph
    print(sparseTimes)
    print(denseTimes)
    plt.yscale("log")

    plt.plot(degrees, sparseTimes, label="Sparse", color="b")
    plt.plot(degrees, denseTimes, label="Dense", color="r")
    plt.xlabel("Degree of Polynomial")
    plt.ylabel("Time (seconds)")
    plt.title("Sparse vs. Dense Polynomial Calculation Time")
    plt.legend()

    plt.savefig("./test/poly.png")


if __name__ == "__main__":
    testGraphDenseVsSparse()
