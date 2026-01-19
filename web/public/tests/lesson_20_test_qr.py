# tests/lesson_20_qr.py
import numpy as np
from dataclasses import dataclass
from tests.consts import *
from src.types import mat
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
from src.vec_prj import vec_prj
from src.mat_prj import mat_prj
from src.ortho import ortho
from src.qr import qr

@dataclass
class Case:
    a: mat

def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        A = random_matrix()
        cases.append(Case(A))
    return cases

def run():
    for c in load_cases():
        ca_copy = copy(c.a)

        Q, R = qr(c.a)
        QM = np.array(Q)
        RM = np.array(R)
        np.testing.assert_allclose(QM @ QM.T, np.eye(len(Q)), atol=UNSTABLE_ZERO)
        np.testing.assert_allclose(QM @ RM, c.a, atol=UNSTABLE_ZERO)
        m, n = np.array(c.a).shape
        assert QM.shape == (m, m)
        assert RM.shape == (m, n)
        assert np.allclose(RM, np.triu(RM), atol=UNSTABLE_ZERO)

        np.testing.assert_equal(ca_copy, c.a, "Vou changed the input a")