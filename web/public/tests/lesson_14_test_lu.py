# tests/lesson_14_lu.py
import numpy as np
from dataclasses import dataclass
from tests.consts import *
from src.errors import SingularError, ShapeMismatchedError

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
from src.types import mat, vec


@dataclass
class LUTestCase:
    A: mat


def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        A = random_matrix(square=True)

        # Skip near-singular
        if abs(np.linalg.det(A)) < ZERO:
            continue

        cases.append(LUTestCase(A))

    return cases


def run():
    for c in load_cases():
        A = c.A
        L, U, P = lu(A)
        Lm = np.array(L)
        Um = np.array(U)
        Pm = np.array(P)

        n = A.shape[0]
        assert Lm.shape == (n, n)
        assert Um.shape == (n, n)
        assert Pm.shape == (n, n)

        # Pm is a permutation matrix
        assert np.allclose(Pm @ Pm.T, np.eye(n), atol=ZERO)
        assert np.allclose(np.sum(Pm, axis=0), 1)
        assert np.allclose(np.sum(Pm, axis=1), 1)

        # L lower triangular, diag 1
        assert np.allclose(np.tril(Lm), Lm, atol=ZERO)
        assert np.allclose(np.diag(Lm), np.ones(n), atol=ZERO)

        # U upper triangular
        assert np.allclose(np.triu(Um), Um, atol=ZERO)

        # Decomposition
        assert np.allclose(Pm @ A, Lm @ Um, atol=ZERO)
