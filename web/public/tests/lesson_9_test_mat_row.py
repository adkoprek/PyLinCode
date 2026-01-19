# tests/lesson_8_mat_row.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
from src.mat_row import mat_row
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
    for c in load_cases():
        cA_copy = copy(c.A)

        np.testing.assert_allclose(mat_row(c.A, c.row_index), c.result, atol=0)

        np.testing.assert_equal(cA_copy, c.A, "You changed the input A")
