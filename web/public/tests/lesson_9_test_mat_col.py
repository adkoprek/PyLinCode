# tests/lesson_9_mat_col.py
import numpy as np
from dataclasses import dataclass
from tests.consts import *
from copy import copy

# --- cumulative imports ---
from src.vec_add import vec_add
from src.vec_scl import vec_scl
from src.vec_dot import vec_dot
from src.vec_len import vec_len
from src.vec_nor import vec_nor
from src.mat_siz import mat_siz
from src.mat_add import mat_add
from src.mat_row import mat_row
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

        np.testing.assert_allclose(mat_col(c.A, c.col_index), c.result, atol=0)

        np.testing.assert_equal(cA_copy, c.A, "You changed the input A")
