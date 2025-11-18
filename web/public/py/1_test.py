from dataclasses import dataclass
from copy import copy
import unittest
import numpy as np


@dataclass
class VecAddTestCase:
    a: vec
    b: vec
    result: vec | Exception
    error: bool = False

def load_vec_add():
    cases = []
    for _ in range(TEST_CASES):
        a = random_vector()
        b = random_vector(a.shape)
        cases.append(VecAddTestCase(a, b, a + b))

    for _ in range(ERROR_TEST_CASES):
        a = random_vector()
        diff_shape = (a.shape[0] + random.randint(1, 10))
        b = random_vector(diff_shape)
        cases.append(VecAddTestCase(a, b, ShapeMismatchedError, error=True))

    return cases

def test_vec_add(test_case: VecAddTestCase):
    if test_case.error:
        with unittest.TestCase.assertRaises(ShapeMismatchedError):
            vec_add(test_case.a, test_case.b)

    else:
        result = vec_add(test_case.a, test_case.b)
        np.testing.assert_allclose(result, test_case.result, atol=0)

def test():
    cases = load_vec_add()
    for case in cases:
        if not cases.error:
            a = copy(cases.a)
            b = copy(cases.b)

        test_vec_add(case)

        if not cases.error:
            assert a == cases.a, "Your solution changed the input vector a"
            assert b == cases.b, "Your solution changed the input vector b"