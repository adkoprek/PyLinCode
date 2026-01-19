# tests/lesson_9_mat_col.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
from src.mat_col import mat_col
from src.types import mat, vec


@dataclass
class Case:
    A: mat
    col_index: int
    result: vec


def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        A = np.random.rand(np.random.randint(1, 5), np.random.randint(1, 5))
        col_index = A.shape[1] - 1
        cases.append(Case(A, col_index, A[:, col_index]))
    return cases


def run():
    for c in load_cases():
        cA_copy = copy(c.A)

        np.testing.assert_allclose(mat_col(c.A.tolist(), c.col_index), c.result, atol=0)

        np.testing.assert_equal(cA_copy, c.A, "You changed the input A")
