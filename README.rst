.. PolyTensor documentation master file, created by
   sphinx-quickstart on Fri Dec 22 09:52:54 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ./docs/source/_static/icon/moonrabbit.svg
  :align: center
  :width: 100
  :alt: polytensor logo 

``polytensor``
==============

``polytensor`` is a python package for CUDA-accelerated, parallel polynomial evaluation and regression.

$$f(x) = c + \\sum_{i=0}^n a_i x_i + \\sum_{i < j}^n a_{i,j} x_i x_j + ... $$

Why?
----

   Evaluating standard, non-matrix polynomials in a CUDA-accelerated, parallel fashion has never been easier! Polytensor is a PyTorch-based package for computing millions of polynomials in parallel on a CUDA GPU. We offer two flavors, sparse-vanilla and dense-rocky-road. My work on quantum-inspired energy models requires computing all kinds of polynomials for optimization and dynamic simulations, and I wanted a clean way of computing the energy function for these models in parallel.


Quick Start
------------

To use the latest stable version of ``polytensor``, install it using ``pip`` from the command line:

.. code-block:: console

   $ pip install polytensor


For the latest development version, install it directly from this repo:

.. code-block:: console

   $ python -m venv .venv
   $ source .venv/bin/activate
   $ (.venv) python -m pip install git+https://github.com/btrainwilson/polytensor.git

Or, if you want to develop ``polytensor``, install it in editable mode:

.. code-block:: console

    $ git clone git+https://github.com/btrainwilson/polytensor.git
    $ python -m pip install -e polytensor


Usage
-----

Examples
--------
All of the following examples assume that you have imported ``polytensor``:

.. code-block:: python

    import polytensor

Polynomial
~~~~~~~~~~~~

The easiest way to begin is by creating a polynomial with a list of terms and coefficients. Consider the following polynomial 



$$f(x) = x_0 + 2 x_1 + 3 x_0 x_1 + 5 x_1^2$$


Each term is specified by a list of indeces and a value. For example, the coefficient $3$ is associated with the term $3 x_0 x_1$, the coefficient $1$ is associated with the term $x_0$, etc.

.. code-block:: python

    terms = {
      (0,)   : 1.0,     # 1.0 * x_0
      (1,)   : 2.0,     # 2.0 * x_1
      (0, 1) : 3.0,     # 3.0 * x_0 * x_1
      (1, 1) : 5.0,     # 5.0 * x_1^2
    }

We can also create random polynomials using the ``polytensor.generators`` module. For example, the following expression is a random polynomial with 10 variables where the coefficients are sampled with a Gaussian $\\mathcal{N}(0,1)$.

$$f(x) = \\sum_{i=0}^{10} a_i x_i + \\sum_{i < j}^{10} a_{i,j} x_i x_j + \\sum_{i < j < k}^{10} a_{i,j, k} x_i x_j x_k + ... : a_s \\sim \\mathcal{N}(0, 1)$$

where $s$ is the term in the polynomial, e.g., $s = (i, j, k)$.


.. code-block:: python

    import torch

    num_vars = 10

    # Create a random polynomial with 10 variables and 5 terms per degree
    num_per_degree = [num_vars, 5, 5, 5]

    # Function to sample coefficients
    sample_fn = lambda: torch.randn(1)


    # Create coefficients for a random polynomial with 10 variables and 5 terms per degree up to degree 4
    terms = polytensor.generators.coeffPUBORandomSampler(
        n=num_vars, num_terms=num_per_degree,sample_fn=sample_fn
        )

Given these coefficients, we can create a polynomial using either a sparse representation or a dense representation. The sparse representation is more efficient for polynomials with fewer terms, while the dense representation is more efficient for polynomials with more terms.


Sparse Polynomials
~~~~~~~~~~~~~~~~~~

Under the hood, the terms remain in their dictionary definition, where the keys are tuples of indeces and the values are the coefficients. For example, the following code creates a sparse polynomial with the coefficients from the previous example.

.. code-block:: python


    terms = {
      (0,)   : 1.0,     # 1.0 * x_0
      (1,)   : 2.0,     # 2.0 * x_1
      (0, 1) : 3.0,     # 3.0 * x_0 * x_1
      (1, 1) : 5.0,     # 5.0 * x_1^2
    }

    poly = polytensor.SparsePolynomial(terms)

    x = torch.Tensor([1.0, 2.0])

    # Evaluate the polynomial at x
    y_p = poly(x)

    # Which is equivalent to
    y_s = 0.0
    for term, v in terms.items():
        y_s = y_s + v * torch.prod(x[..., term])

    assert np.allclose(y_p.detach().cpu().numpy(), y_s.detach().cpu().numpy())

In fact, the loop above is exactly how the polynomial is evaluated. The ``SparsePolynomial`` class is a wrapper around the dictionary of terms and coefficients. The ``__call__`` method loops through the terms and evaluates the polynomial at the given point :math:`x`. Now, we consider dense polynomials.


Dense Polynomials
~~~~~~~~~~~~~~~~~

At a glance, the ``DensePolynomial`` stores the terms in a list of dense ``torch.Tensor`` s, one tensor for each degree, where the indeces of the tensor are the term indeces and the tensor element is the coefficient. The ``DensePolynomial`` class exploits the ``einsum`` function in ``torch`` to evaluate the polynomial using the dense tensors. 


Sparse vs Dense Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When to use the sparse representation? The sparse representation is more efficient than the dense representation when the number of terms :math:`N` is small compared to the number of possible terms, i.e., 

$$N << n^d$$

For ``polytensor.DensePolynomial``, The number of terms in the tensor for degree :math:`d` is :math:`n^d` where :math:`n` is the number of variables in the polynomial. The einsum computation using this representation is way faster than the sparse enumeration if the number of terms is similar to the size of the tensors. Under the hood of ``polytensor.DensePolynomial``, ``torch.einsum`` exploits CUDA acceleration to parallelize the computation. However, if the number of terms in the polynomial is nowhere close to the number of terms in the dense tensor representation, then most of the terms in the dense tensors will be :math:`0` and the sparse polynomial is a better representation. For example, if your polynomial has :math:`100` terms, most of which are quadratic or linear, then a dense representation is likely more efficient. However, if those 100 terms are distributed throughout 6 degree monomials, then a sparse representation is more efficient.

Contributing
------------

We welcome contributions! 

To set up the test environment (.tenv virtual environment), run the following commands:

.. code-block:: console

    $ git clone git+https://github.com/btrainwilson/polytensor.git
    $ cd polytensor
    $ make .tenv
    $ source .tenv/bin/activate

This will handle installing the development dependencies and setting up the virtual environment. 

Testing
~~~~~~~~~~~~~

To run the tests, run the following command:

.. code-block:: console

    $ make test

If everything is set up properly, the tests should pass with green text at the bottom. 

Documentation
~~~~~~~~~~~~~

To build the documentation, run the following command:

.. code-block:: console

    $ make doc

This will build the documentation in the ``docs/build`` directory. 
To view the documentation,  

.. code-block:: console

    $ make serve 

and navigate to `localhost:8018` in your browser.

Pull Requests
~~~~~~~~~~~~~

To submit a contribution, fork the repo and submit a pull request with your changes. We will review the request by running our test suites to ensure the interface is not broken and then check for code cleanliness and correctness. To increase the chances of accepting a PR, build a unit test in the test/ directory as a part of the PR.  


