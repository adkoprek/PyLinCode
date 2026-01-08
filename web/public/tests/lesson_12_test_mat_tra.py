# tests/lesson_12_mat_tra.py
import numpy as np
from dataclasses import dataclass
from tests.consts import *

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
from src.types import mat


@dataclass
class Case:
    A: mat
    result: mat


def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        A = np.random.rand(np.random.randint(1, 5), np.random.randint(1, 5))
        cases.append(Case(A, A.T))
    return cases


def run():
    for c in load_cases():
        np.testing.assert_allclose(mat_tra(c.A), c.result, atol=0)
