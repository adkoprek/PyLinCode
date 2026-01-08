# tests/lesson_17_vec_prj.py
import numpy as np
from dataclasses import dataclass
from tests.consts import *
from src.types import vec

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

@dataclass
class Case:
    a: vec
    b: vec
    result: vec

def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        v = random_vector()
        u = random_vector(v.shape)
        cases.append(Case(u, v, (np.dot(v, u) / np.dot(u, u)) * u))
    return cases

def run():
    for c in load_cases():
        result = vec_prj(c.a, c.b)
        np.testing.assert_allclose(result, c.result, atol=0)
