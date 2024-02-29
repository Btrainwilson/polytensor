import torch
import random
import networkx as nx
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


def denseFromSparse(coeffs: dict, num_bits: int):
    """
    Generates the coefficients for a dense polynomial from a sparse represention.

    Parameters:
        coeffs : Degree of the polynomial
        num_bits : Number of input variables in the polynomial

    Returns:
        List : Tensor representing the high-dimensional triangular polynomial matrix.
    """

    new_coeffs = {}

    for term, value in coeffs.items():
        degree = len(term)

        # Constant special case
        if term == ():
            new_coeffs[degree] = torch.Tensor([value])

        if degree not in new_coeffs:
            new_coeffs[degree] = torch.zeros(*([num_bits] * degree))

        new_coeffs[degree][term] = value

    return [coeff for d, coeff in sorted(new_coeffs.items())]


def dense(n: int, degree: int, device="cuda", dtype=torch.float):
    tensors = [torch.nn.Parameter(torch.rand([1], device=device, dtype=device))]

    for i in range(1, degree + 1):
        tensors.append(
            torch.nn.Parameter(torch.rand(*([n] * i), device=device, dtype=device))
        )

    return tensors


def sparseFromDict(coefficients, dtype, device):
    pass