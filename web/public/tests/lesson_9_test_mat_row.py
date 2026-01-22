# tests/lesson_8_mat_row.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
import src.mat_row as mat_row
from src.types import mat, vec


@dataclass
class Case:
    A: mat
    row_index: int
    result: vec


def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        A = np.random.rand(np.random.randint(1, 5), np.random.randint(1, 5))
        row_index = A.shape[0] - 1
        cases.append(Case(A, row_index, A[row_index, :]))
    return cases


def run():
    globals_before = set(mat_row.__dict__.keys())

    for c in load_cases():
        cA_copy = copy(c.A)

        np.testing.assert_allclose(mat_row.mat_row(c.A.tolist(), c.row_index), c.result, atol=0)

        np.testing.assert_equal(cA_copy, c.A, "You changed the input A")

    globals_after = set(mat_row.__dict__.keys())
    new_globals = globals_after - globals_before
    assert not new_globals, f"You created a global variable {new_globals} which is forbidden"