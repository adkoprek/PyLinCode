# tests/lesson_11_mat_mul.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
from src.errors import ShapeMismatchedError
from src.mat_mul import mat_mul
from src.types import mat


@dataclass
class Case:
    A: mat
    B: mat
    result: mat | Exception
    error: bool = False


def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        A = np.random.rand(np.random.randint(1, 5), np.random.randint(1, 5))
        B = np.random.rand(A.shape[1], np.random.randint(1, 5))
        cases.append(Case(A, B, A @ B))

    for _ in range(ERROR_TEST_CASES):
        A = np.random.rand(np.random.randint(1, 5), np.random.randint(1, 5))
        B = np.random.rand(A.shape[1] + 1, np.random.randint(1, 5))
        cases.append(Case(A, B, ShapeMismatchedError, error=True))

    return cases


def run():
    for c in load_cases():
        if c.error:
            try:
                mat_mul(c.A, c.B)
            except c.result:
                continue
            raise AssertionError("mat_mul: expected ShapeMismatchedError")
        else:
            cA_copy = copy(c.A)
            cB_copy = copy(c.B)

            np.testing.assert_allclose(mat_mul(c.A, c.B), c.result, atol=0)

            np.testing.assert_equal(cA_copy, c.A, "You changed the input A")
            np.testing.assert_equal(cB_copy, c.B, "You changed the input B")
