# tests/lesson_13_mat_vec_mul.py
import numpy as np
from dataclasses import dataclass
from tests.consts import *
from src.errors import ShapeMismatchedError
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
from src.mat_ide import mat_ide
from src.mat_mul import mat_mul
from src.mat_tra import mat_tra
from src.mat_vec import mat_vec_mul
from src.types import mat, vec


@dataclass
class Case:
    a: mat
    v: vec
    result: vec | Exception
    error: bool = False


def load_cases():
    cases = []

    for _ in range(TEST_CASES):
        A = random_matrix()
        x = random_vector(A.shape[1])
        cases.append(Case(A, x, np.matmul(A, x)))

    for _ in range(ERROR_TEST_CASES):
        A = random_matrix()
        x = random_vector(A.shape[1] + np.random.randint(1, 10))
        cases.append(Case(A, x, ShapeMismatchedError, error=True))

    return cases


def run():
    for c in load_cases():
        if c.error:
            try:
                mat_vec_mul(c.a, c.v)
            except c.result:
                continue
            raise AssertionError("mat_vec_mul: expected ShapeMismatchedError")
        else:
            cA_copy = copy(c.A)
            cv_copy = copy(c.v)

            np.testing.assert_allclose(mat_vec_mul(c.a, c.v), c.result, atol=0)

            np.testing.assert_equal(cA_copy, c.A, "You changed the input A")
            np.testing.assert_equal(cv_copy, c.v, "You changed the input v")
