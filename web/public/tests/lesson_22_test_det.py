# tests/lesson_21_det.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
from src.types import mat
from src.errors import ShapeMismatchedError
from src.det import det

@dataclass
class Case:
    a: mat
    result: float | Exception
    error: bool = False

def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        A = random_matrix(square=True)
        cases.append(Case(A, np.linalg.det(A)))

    for _ in range(ERROR_TEST_CASES):
        A = random_matrix()
        if A.shape[0] == A.shape[1]:
            continue
        cases.append(Case(A, ShapeMismatchedError, error=True))

    return cases

def run():
    for c in load_cases():
        if c.error:
            try:
                det(c.a)
            except ShapeMismatchedError:
                continue
            raise AssertionError("det: expected ShapeMismatchedError")
        else:
            ca_copy = copy(c.a)

            result = det(c.a.tolist())
            assert abs(result - c.result) < ZERO

            np.testing.assert_equal(ca_copy, c.a, "Vou changed the input a")
