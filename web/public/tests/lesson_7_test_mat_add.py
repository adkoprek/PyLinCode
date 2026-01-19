# tests/lesson_7_mat_add.py
import numpy as np
from dataclasses import dataclass
from tests.consts import *
from copy import copy

# --- include all previous lessons ---
from src.vec_add import vec_add
from src.vec_scl import vec_scl
from src.vec_dot import vec_dot
from src.vec_len import vec_len
from src.vec_nor import vec_nor
from src.mat_siz import mat_siz
from src.mat_add import mat_add
from src.types import mat
from src.errors import ShapeMismatchedError


@dataclass
class Case:
    A: mat
    B: mat
    result: mat | Exception
    error: bool = False


def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        A = np.random.rand(np.random.randint(1, 5), np.random.randint(1, 5))
        B = np.random.rand(*A.shape)
        cases.append(Case(A, B, A + B))

    for _ in range(ERROR_TEST_CASES):
        A = np.random.rand(np.random.randint(1, 5), np.random.randint(1, 5))
        B = np.random.rand(A.shape[0] + 1, A.shape[1] + 1)
        cases.append(Case(A, B, ShapeMismatchedError, error=True))

    return cases


def run():
    for c in load_cases():
        if c.error:
            try:
                mat_add(c.A, c.B)
            except c.result:
                continue
            raise AssertionError("mat_add: expected ShapeMismatchedError")
        else:
            cA_copy = copy(c.A)
            cB_copy = copy(c.B)

            np.testing.assert_allclose(mat_add(c.A, c.B), c.result, atol=0)

            np.testing.assert_equal(cA_copy, c.A, "You changed the input A")
            np.testing.assert_equal(cB_copy, c.B, "You changed the input B")
