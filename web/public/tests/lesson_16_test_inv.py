# tests/lesson_16_inv.py
import numpy as np
from dataclasses import dataclass
from tests.consts import *
from src.errors import SingularError
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
from src.types import mat


@dataclass
class Case:
    a: mat
    result: mat | Exception
    error: bool = False


def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        A = random_matrix(square=True)

        if abs(np.linalg.det(A)) < ZERO:
            continue

        cases.append(Case(A, np.linalg.inv(A)))

    for _ in range(ERROR_TEST_CASES):
        A = random_matrix(square=True)
        first_row = np.random.randint(0, A.shape[0] - 1)
        second_row = first_row + 1
        A[first_row, :] = A[second_row, :]
        assert abs(np.linalg.det(A)) < ZERO
        cases.append(Case(A, SingularError, error=True))

    return cases


def run():
    for c in load_cases():
        if c.error:
            try:
                inv(c.a)
            except SingularError:
                continue
            raise AssertionError("inv: expected SingularError")
        else:
            cA_copy = copy(c.A)

            result = inv(c.a)
            np.testing.assert_allclose(result, c.result, atol=ZERO)

            np.testing.assert_equal(cA_copy, c.A, "You changed the input A")
