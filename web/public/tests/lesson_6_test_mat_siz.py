# tests/lesson_6_mat_siz.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
import src.mat_siz as mat_siz
from src.types import mat


@dataclass
class Case:
    A: mat
    result: tuple[int, int]


def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        rows = np.random.randint(1, 10)
        cols = np.random.randint(1, 10)
        A = np.random.rand(rows, cols)
        cases.append(Case(A, A.shape))
    return cases


def run():
    globals_before = set(mat_siz.__dict__.keys())

    for c in load_cases():
        cA_copy = copy(c.A)

        result = mat_siz.mat_siz(c.A.tolist())
        assert result == c.result

        np.testing.assert_equal(cA_copy, c.A, "You changed the input A")

    globals_after = set(mat_siz.__dict__.keys())
    new_globals = globals_after - globals_before
    assert not new_globals, f"You created a global variable {new_globals} which is forbidden"