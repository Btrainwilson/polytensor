import torch


def denseFromSparse(coeffs: dict, sample_fn: callable):
    """
    Generates the coefficients for a dense polynomial from a sparse represention.

    Parameters:
        coeffs : Degree of the polynomial

    Returns:
        List : Tensor representing the high-dimensional triangular polynomial matrix.
    """

    n = 0
    deg = 0
    for k, v in coeffs.items():
        deg = max(len(k), deg)
        n = max(n, max(k))
    n += 1

    new_coeffs = [torch.nn.Parameter(torch.zeros(1))]

    for i in range(1, deg + 1):
        new_coeffs.append(torch.zeros(*([n] * i)))

    for k, v in coeffs.items():
        new_coeffs[len(k)][k] = v

    return new_coeffs


def dense(n: int, degree: int):
    tensors = [torch.nn.Parameter(torch.rand([1]))]

    for i in range(1, degree + 1):
        tensors.append(torch.nn.Parameter(torch.rand(*([n] * i))))

    return tensors
