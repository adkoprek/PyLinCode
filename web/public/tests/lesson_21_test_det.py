# tests/lesson_21_det.py
import numpy as np
from dataclasses import dataclass
from tests.consts import *
from src.types import mat
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
from src.lu import lu
from src.solve import solve
from src.inverse import inv
from src.determinant import det

@dataclass
class Case:
    a: mat
    result: float | Exception
    error: bool = False

def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        A = random_matrix(square=True)
        cases.append(Case(A, np.linalg.det(A)))

    for _ in range(ERROR_TEST_CASES):
        A = random_matrix()
        if A.shape[0] == A.shape[1]:
            continue
        cases.append(Case(A, ShapeMismatchedError, error=True))

    return cases

def run():
    for c in load_cases():
        if c.error:
            try:
                det(c.a)
            except ShapeMismatchedError:
                continue
            raise AssertionError("det: expected ShapeMismatchedError")
        else:
            ca_copy = copy(c.a)

            result = det(c.a)
            assert abs(result - c.result) < ZERO

            np.testing.assert_equal(ca_copy, c.a, "Vou changed the input a")
