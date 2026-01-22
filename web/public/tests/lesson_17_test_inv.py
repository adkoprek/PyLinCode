# tests/lesson_16_inv.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
from src.errors import SingularError
import src.inv as inv
from src.types import mat


@dataclass
class Case:
    a: mat
    result: mat | Exception
    error: bool = False


def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        A = random_matrix(square=True)

        if abs(np.linalg.det(A)) < ZERO:
            continue

        cases.append(Case(A, np.linalg.inv(A)))

    for _ in range(ERROR_TEST_CASES):
        A = random_matrix(square=True)
        first_row = np.random.randint(0, A.shape[0] - 1)
        second_row = first_row + 1
        A[first_row, :] = A[second_row, :]
        assert abs(np.linalg.det(A)) < ZERO
        cases.append(Case(A, SingularError, error=True))

    return cases


def run():
    globals_before = set(inv.__dict__.keys())

    for c in load_cases():
        if c.error:
            try:
                inv.inv(c.a)
            except SingularError:
                continue
            raise AssertionError("inv: expected SingularError")
        else:
            ca_copy = copy(c.a)

            result = inv.inv(c.a.tolist())
            np.testing.assert_allclose(result, c.result, atol=ZERO)

            np.testing.assert_equal(ca_copy, c.a, "You changed the input a")

    globals_after = set(inv.__dict__.keys())
    new_globals = globals_after - globals_before
    assert not new_globals, f"You created a global variable {new_globals} which is forbidden"