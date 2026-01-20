# tests/lesson_8_mat_scl.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
from src.mat_scl import mat_scl
from src.types import mat


@dataclass
class Case:
    A: mat
    S: int
    result: mat | Exception
    error: bool = False


def load_cases():
    cases = []

    for _ in range(TEST_CASES):
        A = np.random.rand(np.random.randint(1, 5), np.random.randint(1, 5))
        S = np.random.randint(-10, 11)
        cases.append(Case(A, S, A * S))

    return cases


def run():
    for c in load_cases():
        if c.error:
            try:
                mat_scl(c.A, c.S)
            except c.result:
                continue
            raise AssertionError("mat_scl: expected error")
        else:
            cA_copy = copy(c.A)

            np.testing.assert_allclose(mat_scl(c.A.tolist(), c.S), c.result, atol=0)

            np.testing.assert_equal(cA_copy, c.A, "You changed the input A")
