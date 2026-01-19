# tests/lesson_15_solve.py
import numpy as np
from dataclasses import dataclass
from tests.consts import *
from src.errors import SingularError, ShapeMismatchedError
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
from src.types import mat, vec


@dataclass
class SolveTestCase:
    A: mat
    b: vec
    error: None | Exception = None


def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        A = random_matrix(square=True)
        b = random_vector(A.shape[0])
        if abs(np.linalg.det(A)) < ZERO:
            continue
        cases.append(SolveTestCase(A, b))

    for _ in range(ERROR_TEST_CASES):
        # singular
        A = random_matrix(square=True)
        b = random_vector(A.shape[0])
        first_row = np.random.randint(0, A.shape[0] - 1)
        second_row = first_row + 1
        A[first_row, :] = A[second_row, :]
        assert abs(np.linalg.det(A)) < ZERO
        cases.append(SolveTestCase(A, b, SingularError))

    for _ in range(ERROR_TEST_CASES):
        # shape mismatch
        A = random_matrix()
        b = random_vector(A.shape[0])
        if A.shape[0] == A.shape[1]:
            continue
        cases.append(SolveTestCase(A, b, ShapeMismatchedError))

    return cases


def run():
    for c in load_cases():
        if c.error == SingularError:
            try:
                solve(c.A, c.b)
            except SingularError:
                continue
            raise AssertionError("solve: expected SingularError")
        elif c.error == ShapeMismatchedError:
            try:
                solve(c.A, c.b)
            except ShapeMismatchedError:
                continue
            raise AssertionError("solve: expected ShapeMismatchedError")
        else:
            cA_copy = copy(c.A)
            cb_copy = copy(c.b)

            x = solve(c.A, c.b)
            A = np.asarray(c.A, dtype=float)
            x = np.asarray(x, dtype=float)
            b = np.asarray(c.b, dtype=float)
            m, n = A.shape
            assert x.shape == (n,)
            assert b.shape == (m,)
            np.testing.assert_allclose(A @ x, b, atol=ZERO)

            np.testing.assert_equal(cA_copy, c.A, "You changed the input A")
            np.testing.assert_equal(cb_copy, c.b, "You changed the input b")
