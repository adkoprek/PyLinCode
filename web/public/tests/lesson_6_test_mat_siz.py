# tests/lesson_6_mat_siz.py
import numpy as np
from dataclasses import dataclass
from tests.consts import *
from copy import copy

# --- include ALL previous lessons (vectors) ---
from src.vec_add import vec_add
from src.vec_scl import vec_scl
from src.vec_dot import vec_dot
from src.vec_len import vec_len
from src.vec_nor import vec_nor

# --- current lesson ---
from src.mat_siz import mat_siz
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
    for c in load_cases():
        cA_copy = copy(c.A)

        result = mat_siz(c.A)
        assert result == c.result

        np.testing.assert_equal(cA_copy, c.A, "You changed the input A")
