import torch
import random
from beartype import beartype
from beartype.typing import List, Callable, Union
from collections.abc import Iterable


@beartype
def random_combination_generator(iterable: Iterable, k: int):
    """
    Generates random, unique combinations of size k from the iterable.

    Source: https://stackoverflow.com/questions/22229796/choose-at-random-from-combinations

    Args:
        Iterable : iterable
        int : k

    Returns:
        generator : Each element is a tuple of
    """
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), k))
    return tuple(pool[i] for i in indices)


@beartype
def coeffRandomSampler(
    n: int, num_terms: int, degree: int, sample_fn: Callable
) -> dict:
    """
    Generates random non-zero terms of a polynomial.

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
        terms[random_combination_generator(list(range(n)), degree)] = sample_fn()
    return terms


def coeffPUBORandomSampler(n: int, num_terms: List[int], sample_fn: Callable):
    """
    Generates random non-zero terms of a polynomial using a filling factor.

    Args:
        n : Dimension of the polynomial
        num_terms : Number of non-zero terms in the polynomial for each degree
        degree : Degree of the polynomial
        sample_fn : Function to sample the coefficients of the polynomial

    Returns:
        Dict : Tensor representing the high-dimensional triangular polynomial matrix.
    """
    terms = {}
    for i, nt in enumerate(num_terms):
        terms.update(
            coeffRandomSampler(n=n, num_terms=nt, degree=i + 1, sample_fn=sample_fn)
        )
    return terms


def denseFromSparse(
    coeffs: dict, num_bits: Union[int, None] = None, device="cpu", dtype=torch.float
):
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

    if num_bits:
        n = num_bits

    new_coeffs = [torch.nn.Parameter(torch.zeros(1, dtype=dtype, device=device))]

    for i in range(1, deg + 1):
        new_coeffs.append(torch.zeros(*([n] * i), dtype=dtype, device=device))

    for k, v in coeffs.items():
        new_coeffs[len(k)][k] = v

    return new_coeffs


def dense(n: int, degree: int, device="cuda", dtype=torch.float):
    tensors = [torch.nn.Parameter(torch.rand([1], device=device, dtype=device))]

    for i in range(1, degree + 1):
        tensors.append(
            torch.nn.Parameter(torch.rand(*([n] * i), device=device, dtype=device))
        )

    return tensors
