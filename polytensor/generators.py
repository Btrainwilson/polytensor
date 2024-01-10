import torch
import random
import networkx as nx
from beartype import beartype
from typing import List, Callable, Union
from collections.abc import Iterable


def from_networkx(g: nx.Graph, key="weight"):
    """
    Converts a networkx graph to a dictionary of terms for a polynomial.

    Args:
        g : Networkx graph
        key : Attribute to use for the value of the term
    """
    terms = {}
    for node in list(g.nodes(data=key)):
        if node[1] is None:
            raise ValueError(
                f"Node {node} must have non-None value for attribute {key}"
            )
        if node[1] != 0:
            terms[tuple([node[0]])] = node[1]

    # Convert edges
    for edge in nx.to_edgelist(g):
        if edge[1] != 0:
            terms[edge[:2]] = edge[2]

    return terms


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


def denseFromSparse(coeffs: dict, num_bits: Union[int, None] = None):
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
