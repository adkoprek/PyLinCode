# tests/lesson_15_solve.py import numpy as np

import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
from src.errors import SingularError, ShapeMismatchedError
from src.solve import solve, for_sub, bck_sub
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
            # ---------------- solve ----------------
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

            # ---------------- for_sub ----------------
            n = np.random.randint(1, 6)
            L = np.tril(np.random.rand(n, n))
            np.fill_diagonal(L, np.random.rand(n) + 1.0)  # avoid zeros on diagonal
            b_f = np.random.rand(n)

            L_copy = copy(L)
            b_f_copy = copy(b_f)

            y = for_sub(L, b_f)
            y = np.asarray(y, dtype=float)

            assert y.shape == (n,)
            np.testing.assert_allclose(L @ y, b_f, atol=ZERO)

            np.testing.assert_equal(L_copy, L, "for_sub changed input L")
            np.testing.assert_equal(b_f_copy, b_f, "for_sub changed input b")

            # ---------------- bck_sub ----------------
            U = np.triu(np.random.rand(n, n))
            np.fill_diagonal(U, np.random.rand(n) + 1.0)  # avoid zeros on diagonal
            b_b = np.random.rand(n)

            U_copy = copy(U)
            b_b_copy = copy(b_b)

            x = bck_sub(U, b_b)
            x = np.asarray(x, dtype=float)

            assert x.shape == (n,)
            np.testing.assert_allclose(U @ x, b_b, atol=ZERO)

            np.testing.assert_equal(U_copy, U, "bck_sub changed input U")
            np.testing.assert_equal(b_b_copy, b_b, "bck_sub changed input b")
