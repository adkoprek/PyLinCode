# tests/lesson_8_mat_row.py
import numpy as np
from dataclasses import dataclass
from tests.consts import *

# --- cumulative imports ---
from src.vec_add import vec_add
from src.vec_scl import vec_scl
from src.vec_dot import vec_dot
from src.vec_len import vec_len
from src.vec_nor import vec_nor
from src.mat_siz import mat_siz
from src.mat_add import mat_add
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
        np.testing.assert_allclose(mat_row(c.A, c.row_index), c.result, atol=0)
