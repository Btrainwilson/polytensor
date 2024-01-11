.. PolyTensor documentation master file, created by
   sphinx-quickstart on Fri Dec 22 09:52:54 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``polytensor``
==============

``polytensor`` is a python package for CUDA-accelerated, parallel polynomial evaluation and regression.

.. math::

   f(x) = c + \sum_{i=0}^n a_i x_i + \sum_{i < j}^n a_{i,j} x_i x_j + ... 

API
---
.. toctree::
   :maxdepth: 2

   polytensor


Usage
-----

Installation
~~~~~~~~~~~~

To use ``polytensor``, first install it using ``pip`` from the command line:

.. code-block:: console


   $ python -m venv .venv
   $ source .venv/bin/activate
   $ (.venv) python -m pip install git+https://github.com/btrainwilson/polytensor.git

Or, unzip the package and install it using ``pip`` from the command line:

.. code-block:: console

    $ python -m venv .venv
    $ source .venv/bin/activate
    $ (.venv) unzip polytensor.zip
    $ (.venv) cd polytensor
    $ (.venv) python -m pip install -e .

Examples
--------
All of the following examples assume that you have imported ``polytensor``:

.. code-block:: python

    import polytensor

Polynomial
~~~~~~~~~~~~

The easiest way to begin is by creating a polynomial with a list of terms and coefficients. Consider the following polynomial 


.. math::

   f(x) = x_0 + 2 x_1 + 3 x_0 x_1 + 5 x_1^2


Each term is specified by a list of indeces and a value. For example, the coefficient ``3`` is associated with the term ``3 x_0 x_1``, the coefficient ``1`` is associated with the term ``x_0``, etc.

.. code-block:: python

    terms = {
      tuple((0)) : 1.0,        # 1.0 * x_0
      tuple((1)) : 2.0,        # 2.0 * x_1
      (0, 1) : 3.0,     # 3.0 * x_0 * x_1
      (1, 1) : 5.0,     # 5.0 * x_1^2
    }

We can also create random polynomials using the ``polytensor.generators`` module. For example, the following code creates a random polynomial with 10 variables by sampling a Gaussian :math:`\mathcal{N}(0,1)`.

.. math::

   f(x) = \sum_{i=0}^{10} a_i x_i + \sum_{i < j}^{10} a_{i,j} x_i x_j + \sum_{i < j < k}^{10} a_{i,j, k} x_i x_j x_k + ... : a_s \sim \mathcal{N}(0, 1)

where :math:`s` is the term in the polynomial, e.g., :math:`s = (i, j, k)`.


.. code-block:: python

    import torch

    num_vars = 10

    # Create a random polynomial with 10 variables and 5 terms per degree
    num_per_degree = [num_vars, 5, 5, 5]

    # Function to sample coefficients
    sample_fn = lambda: torch.rand(1)


    coefficients = polytensor.generators.coeffPUBORandomSampler(
        n=num_vars, num_terms=num_per_degree,sample_fn=sample_fn
        )

Given these coefficients, we can create a polynomial using either a sparse representation or a dense representation. The sparse representation is more efficient for polynomials with fewer terms, while the dense representation is more efficient for polynomials with more terms.


Sparse Polynomials
~~~~~~~~~~~~~~~~~~

Under the hood, the terms remain in their dictionary definition, where the keys are tuples of indeces and the values are the coefficients. For example, the following code creates a sparse polynomial with the coefficients from the previous example.

.. code-block:: python


    terms = {
      tuple((0)) : 1.0,       # 1.0 * x_0
      tuple((1)) : 2.0,       # 2.0 * x_1
      (0, 1) : 3.0,           # 3.0 * x_0 * x_1
      (1, 1) : 5.0,           # 5.0 * x_1^2
    }

    poly = polytensor.SparsePolynomial(coefficients)

    x = torch.Tensor([1.0, 2.0])

    # Evaluate the polynomial at x
    y_p = poly(x)

    # Which is equivalent to
    y_s = 0.0

    for term, v in terms.items():
        y_s += v * torch.prod(x[..., term])

    assert np.allclose(y_p.detach().cpu().numpy(), y_s.detach().cpu().numpy())

In fact, the loop above is exactly how the polynomial is evaluated. The ``SparsePolynomial`` class is a wrapper around the dictionary of terms and coefficients. The ``__call__`` method loops through the terms and evaluates the polynomial at the given point ``x``. Now, we consider dense polynomials.


Dense Polynomials
~~~~~~~~~~~~~~~~~

At a glance, the ``DensePolynomial`` stores the terms in a list of dense ``torch.Tensor`` s, one tensor for each degree, where the indeces of the tensor are the term indeces and the tensor element is the coefficient. The ``DensePolynomial`` class exploits the ``einsum`` function in ``torch`` to evaluate the polynomial using the dense tensors. 


Sparse vs Dense Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When to use the sparse representation? The sparse representation is more efficient than the dense representation when the number of terms :math:`N` is small compared to the number of possible terms, i.e., 

.. math::
    N << n^d

For ``polytensor.DensePolynomial``, The number of terms in the tensor for degree :math:`d` is :math:`n^d` where :math:`n` is the number of variables in the polynomial. The einsum computation using this representation is way faster than the sparse enumeration if the number of terms is similar to the size of the tensors. Under the hood of ``polytensor.DensePolynomial``, ``torch.einsum`` exploits CUDA acceleration to parallelize the computation. However, if the number of terms in the polynomial is nowhere close to the number of terms in the dense tensor representation, then most of the terms in the dense tensors will be :math:`0` and the sparse polynomial is a better representation. For example, if your polynomial has :math:`100` terms, most of which are quadratic or linear, then a dense representation is likely more efficient. However, if those 100 terms are distributed throughout 6 degree monomials, then a sparse representation is more efficient.
