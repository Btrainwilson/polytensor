import torch
import random
import itertools
from beartype import beartype


def random_combination(iterable, r):
    """Random selection from itertools.combinations(iterable, r).
    From: https://stackoverflow.com/questions/22229796/choose-at-random-from-combinations
    """
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


@beartype
def randomConstant(n: int, num_terms: int, degree: int, sample_fn) -> dict:
    """
    Generates random non-zero terms of a polynomial using a filling factor.

    Parameters:
        n : Dimension of the polynomial
        num_terms : Number of non-zero terms in the polynomial
        degree : Degree of the polynomial
        sample_fn : Function to sample the coefficients of the polynomial

    Returns:
        Dict : Tensor representing the high-dimensional triangular polynomial matrix.
    """
    terms = {}
    while len(terms) < num_terms:
        terms[random_combination(list(range(n)), degree)] = sample_fn()
    return terms


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
