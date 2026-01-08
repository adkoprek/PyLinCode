# tests/lesson_18_mat_prj.py
import numpy as np
from dataclasses import dataclass
from tests.consts import *
from src.types import mat
from src.errors import SingularError

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

@dataclass
class Case:
    a: mat
    result: mat

def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        A = random_matrix()
        ATA = A.T @ A
        if abs(np.linalg.det(ATA)) < ZERO:
            continue
        cases.append(Case(A, A @ np.linalg.inv(ATA) @ A.T))
    return cases

def run():
    for c in load_cases():
        result = mat_prj(c.a)
        np.testing.assert_allclose(result, c.result, atol=UNSTABLE_ZERO)
