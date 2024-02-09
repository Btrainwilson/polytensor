###############################################################################
# File: polynomial.py                                                         #
# Author: Blake Wilson                                                        #
# Date: 2024-01-02                                                            #
#                                                                             #
# Polynomial Representation                                                   #
#                                                                             #
###############################################################################

import torch
from torch import nn
import numpy as np
import funcy
from beartype import beartype
from beartype.typing import Union


class SparsePolynomial(torch.nn.Module):
    """
    A sparse autograd differentiable polynomial function.

    :math:`f(x) = \\sum_{s \\in S}\\sigma_s\\prod_{i \\in s}x_i`

    Args:

        coefficients: :math:`s` Stored as a dictionary. Each key represents the monomial variable indices and the value is the coefficient for the term.

    """

    @beartype
    def __init__(
        self,
        coefficients: dict[tuple[int], Union[complex, float, int]],
    ):
        super().__init__()

        # Store the coefficients as a dictionary of parameters.
        # The key is a str representation of the monomial variables and the value is the coefficient.
        # This is done to ensure that the parameters are registered with the module.
        # This is necessary for the autograd and storage to work properly.
        if not () in coefficients:
            coefficients[()] = 0

        self.terms = nn.ParameterDict(
            {
                str(tuple(sorted(k))): nn.Parameter(torch.Tensor([v]))
                for k, v in coefficients.items()
            }
        )

        self.coefficients = coefficients

    def forward(self, x):
        """
        Computes the polynomial function specified by the polynomial coefficients and the input tensor x.

        Args:
            coefficients : dict Each key represents the monomial variables and the value is the coefficient.

            x : torch.Tensor Batched input tensor of shape BxS1x...N.

        Returns:
            torch.Tensor : The value of the polynomial function.
        """
        sum = self.coefficients[()]

        # Scales as O(n*d) where n is the number of terms and d is the degree of the term
        for key, v in self.coefficients.items():
            sum = sum + v * torch.prod(x[..., key], dim=-1, keepdim=True)

        return sum

    def __repr__(self):
        return f"SparseCoefficients({self.coefficients})"


class DensePolynomial(torch.nn.Module):
    """
    A dense autograd differentiable polynomial function.
    Consumes O(n^d) memory where n is the number of terms and d is the degree of the term.

    :math:`f(x) = \\sum_{s \\in S}\\sigma_s\\prod_{i \\in s}x_i`

    Args:
        coefficients: :math:`s` Stored as a list of tensors.

    """

    @beartype
    def __init__(
        self,
        coefficients: list[torch.Tensor],
    ):
        super().__init__()
        self.coefficients = nn.ParameterList([nn.Parameter(c) for c in coefficients])

    def forward(self, x):
        """
        Computes the polynomial function specified by the polynomial coefficients and the input tensor x.

        Args:
            x: Batched input tensor of shape :math:`BxS1x...N`.

        Returns:
            torch.Tensor : The value of the polynomial function.
        """
        s = 0

        for t in self.coefficients:
            s = s + self.polyD(x, t)

        return s

    def polyD(self, x, T):
        """
        Computes the polynomial given by tensor T acting on input vector x
        We use the Einstein summation convention to compute the polynomial.


        Parameters:
            x (torch.Tensor) : Tensor of shape (batch_size, num_dim) representing the configuration of the system.
            T (torch.Tensor) : Tensor representing the high-dimensional triangular polynomial matrix.

        Returns:
            torch.Tensor : The energy for each configuration in the batch.
        """

        k = len(T.shape)

        # Constant term
        if T.shape == torch.Size([]):
            return T.unsqueeze(0)

        params = [T, list(range(k))]

        # Exploit einsum's broadcasting to compute the polynomial
        for i in range(k):
            params.append(x)
            params.append([..., i])

        return torch.einsum(*params).unsqueeze(-1)

    def validate(self):
        """
        Checks if the coefficients are valid for a dense polynomial.
        """

        coefficients = self.coefficients

        if len(coefficients) == 0:
            raise ValueError("Coefficients cannot be empty.")

        for d in range(2, len(coefficients)):
            if coefficients[d] is None:
                continue

            terms = torch.nonzero(coefficients[d]).squeeze()

            # Check if only one term exists
            if terms.dim() == 1:
                terms = terms.unsqueeze(0)

            for i in range(terms.shape[0]):
                if (np.diff(terms[i].cpu().numpy()) < 0).all():
                    raise ValueError(
                        f"Coefficients {terms[i].numpy()} must be in non-decreasing order."
                    )

        return True

    def __repr__(self):
        return f"DensePolynomial(degree={len(self.coefficients)-1})"


class PottsModel(torch.nn.Module):
    @beartype
    def __init__(
        self,
        coefficients: dict[list[int], Union[complex, float, int, torch.Tensor]],
    ):
        super().__init__()

        self.terms = nn.ParameterDict(
            {
                str(tuple(sorted(k))): nn.Parameter(torch.Tensor([v]))
                for k, v in coefficients.items()
            }
        )

        self.coefficients = coefficients

    def forward(self, x):
        sum = 0.0

        for key, v in self.coefficients.items():
            sum = sum + v * torch.eq(
                torch.max(x[..., key], -1).values, torch.min(x[..., key], -1).values
            )

        return sum


class PottsModelOneHot(torch.nn.Module):
    @beartype
    def __init__(
        self,
        coefficients: dict[list[int], Union[complex, float, int, torch.Tensor]],
    ):
        super().__init__()

        self.terms = nn.ParameterDict(
            {
                str(tuple(sorted(k))): nn.Parameter(torch.Tensor([v]))
                for k, v in coefficients.items()
            }
        )

        self.coefficients = coefficients

    def forward(self, x):
        sum = 0.0

        x_dim = len(x.shape) - 2

        for key, v in self.coefficients.items():
            sum = sum + v * torch.sum(torch.prod(x[..., key, :], dim=x_dim), dim=x_dim)

        return sum
