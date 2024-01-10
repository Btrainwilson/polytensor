###############################################################################
# File: polynomial.py                                                         #
# Author: Blake Wilson                                                        #
# Date: 2024-01-02                                                            #
#                                                                             #
# Polynomial Representation                                                   #
#                                                                             #
###############################################################################

import torch
import numpy as np
from beartype import beartype
from beartype.typing import List
from typing import Union
from enum import Enum
from abc import ABC, abstractmethod
import logging


class Polynomial(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self):
        raise NotImplementedError("Polynomial is an abstract class.")

    @abstractmethod
    def validate(self):
        """
        Validates the coefficients.
        """
        raise NotImplementedError("Polynomial is an abstract class.")


class SparsePolynomial(Polynomial):
    """
    A Sparse PyTorch autograd differentiable polynomial function.

    :math:`f(x) = \sum_{s \in S}\sigma_s\prod_{i \in s}x_i`

    Args:

        coefficients: :math:`s` Stored as a dictionary. Each key represents the monomial variable indices and the value is the coefficient for the term.

    """

    @beartype
    def __init__(
        self,
        coefficients: dict[List[int], Union[complex, float, int, torch.Tensor]],
        device: str = "cpu",
        dtype=torch.float,
        **kwargs,
    ):
        super().__init__()

        self.coefficients = coefficients
        self.device = device
        self.dtype = dtype
        self.validate()

    def forward(self, x):
        """
        Computes the polynomial function specified by the polynomial coefficients and the input tensor x.

        Args:
            coefficients: Dictionary of coefficients. Each key represents the monomial variables and the value is the coefficient.

            x: The input to the polynomial function.

        Returns:
            torch.Tensor : The value of the polynomial function.
        """
        r = False
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            r = True

        sum = 0
        # Scales as O(n*d) where n is the number of terms and d is the degree of the term
        for key in self.coefficients:
            sum += self.coeff_vector[self.coeff_map[key]] * torch.prod(x[:, key])

        if r:
            return sum.squeeze()

        return sum

    def validate(self):
        """
        Checks if the coefficients are valid for a sparse polynomial.
        """

        if len(self.coefficients) == 0:
            raise ValueError("Coefficients cannot be empty.")

        for term, value in self.coefficients.items():
            if type(value) not in [
                int,
                float,
                complex,
                torch.Tensor,
                np.ndarray,
            ]:
                raise TypeError(
                    "Coefficients must be a number, numpy array, or a tensor."
                )
            for t in zip(
                [int, float, complex], [torch.int, torch.float, torch.complex]
            ):
                if type(value) == t[0]:
                    if self.dtype != t[1]:
                        logging.warning(
                            f"Coefficient {term} is type {type(value)} and will be converted to type {self.dtype}."
                        )

            if len(term) > 1 and (np.diff(term) < 0).all():
                raise ValueError(
                    f"Coefficients {np.diff(term)} must be in non-decreasing order."
                )

        # Make sure all coefficients are unique
        if len(self.coefficients.keys()) != len(set(self.coefficients.keys())):
            raise ValueError("Coefficients must be unique.")

        self.coeff_vector = torch.nn.ParameterList()
        self.coeff_map = {}
        for i, term in enumerate(self.coefficients.keys()):
            self.coeff_map[term] = i
            if type(value) not in [torch.Tensor]:
                self.coeff_vector.append(
                    torch.nn.Parameter(
                        torch.tensor(self.coefficients[term], dtype=self.dtype)
                    )
                )
            else:
                self.coeff_vector.append(torch.nn.Parameter(self.coefficients[term]))

        return True

    def __repr__(self):
        return f"SparseCoefficients({self.coefficients})"


class DensePolynomial(Polynomial):
    """
    A Dense PyTorch autograd differentiable polynomial function. Uses einsum to compute the polynomial function.

    :math:`f(x) = \sum_{s \in S}\sigma_s\prod_{i \in s}x_i`

    Args:
        coefficients: :math:`s` Stored as a list of tensors.

    """

    @beartype
    def __init__(
        self,
        coefficients: list[torch.Tensor],
        device: str = "cpu",
        dtype=torch.float,
        **kwargs,
    ):
        super().__init__()

        self.coefficients = coefficients
        self.device = device
        self.dtype = dtype
        self.validate()

    def forward(self, x):
        """
        Computes the polynomial function specified by the polynomial coefficients and the input tensor x.

        Args:
            coefficients: Dictionary of coefficients. Each key represents the monomial variables and the value is the coefficient.
            x: The input to the polynomial function.

        Returns:
            torch.Tensor : The value of the polynomial function.
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        s = self.polyD(x, self.coefficients[0])

        if len(self.coefficients) > 1:
            for t in self.coefficients[1:]:
                s = s + self.polyD(x, t)

        return s

    def polyD(self, x, T):
        """
        Computes the polynomial given by tensor T acting on input vector x

        We use the Einstein summation convention to compute the polynomial.
        For example, if we given the 3-dimensional polynomial


        Parameters:
            x (torch.Tensor) : Tensor of shape (batch_size, num_dim) representing the configuration of the system.
            T (torch.Tensor) : Tensor representing the high-dimensional triangular polynomial matrix.

        Returns:
            torch.Tensor : The energy for each configuration in the batch.
        """
        k = len(T.shape)

        params = [T, list(range(k))]

        # Exploit einsum's broadcasting to compute the polynomial
        for i in range(k):
            params.append(x)
            params.append([..., i])

        return torch.einsum(*params)

    def validate(self):
        """
        Checks if the coefficients are valid for a dense polynomial.
        """

        coefficients = self.coefficients

        if len(coefficients) == 0:
            raise ValueError("Coefficients cannot be empty.")

        for d in range(2, len(coefficients)):
            terms = torch.nonzero(coefficients[d]).squeeze()

            for i in range(terms.shape[0]):
                if (np.diff(terms[i].numpy()) < 0).all():
                    raise ValueError(
                        f"Coefficients {terms[i].numpy()} must be in non-decreasing order."
                    )

        return True

    def __repr__(self):
        return f"DensePolynomial(degree={len(self.coefficients)-1})"
