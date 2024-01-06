###############################################################################
# File: coefficients.py                                                       #
# Author: Blake Wilson                                                        #
# Date: 2024-01-02                                                            #
#                                                                             #
# Polynomial Coefficients                                                     #
#                                                                             #
###############################################################################

from beartype import beartype
from abc import ABC, abstractmethod


class Coefficients(ABC):
    def __init__(self):
        raise NotImplementedError("Coefficients is an abstract class.")

    @abstractmethod
    def validate(self):
        """
        Validates the coefficients.
        """
        pass


class SparseCoefficients(Coefficients):
    """
    Sparse polynomial coefficients.
    """

    @beartype
    def __init__(
        self,
        coefficients: dict,
    ):
        super().__init__()

        self.coefficients = coefficients


class DenseCoefficients(Coefficients):
    """
    Dense tensor polynomial coefficients.
    """

    @beartype
    def __init__(
        self,
        coefficients: Union[dict, List],
    ):
        super().__init__()

        self.coefficients = coefficients
