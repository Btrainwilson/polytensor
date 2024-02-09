import polytensor


def testSparseValidation():
    coefficients = polytensor.generators.coeffPUBORandomSampler(
        5, [5, 5, 5, 5], lambda: 1
    )

    polytensor.validate.sparse_coefficients(coefficients)


def testSparseValidationFail():
    test_coeff = [
        {},
        {"a": 6},
        {(1, 2, 3): "h"},
        {("g"): 6},
    ]

    for coefficients in test_coeff:
        try:
            polytensor.validate.sparse_coefficients(coefficients)
        except ValueError:
            pass
        else:
            assert False, f"{coefficients} should raise ValueError."


def testSparseValidationFail():
    test_coeff = [
        {},
        {"a": 6},
        {(1, 2, 3): "h"},
        {("g"): 6},
    ]

    for coefficients in test_coeff:
        try:
            polytensor.validate.sparse_coefficients(coefficients)
        except ValueError:
            pass
        else:
            assert False, f"{coefficients} should raise ValueError."
