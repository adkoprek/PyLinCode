# tests/lesson_19_ortho.py
import numpy as np
from dataclasses import dataclass
from random import randint
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
from src.mat_prj import mat_prj
from src.ortho import ortho

@dataclass
class Case:
    existing: list[vec]
    new: vec

def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        m = randint(2, 10)
        A = random_matrix((m, m))
        if abs(np.linalg.det(A)) < ZERO:
            continue
        Q, _ = np.linalg.qr(A)
        vecs = [list(Q[:, i]) for i in range(Q.shape[1]-1)]
        a = list(A[:, -1])
        cases.append(Case(vecs, a))
    return cases

def run():
    for c in load_cases():
        orthogonalized = ortho(c.existing, c.new)
        bv = np.array(orthogonalized)
        assert abs(np.linalg.norm(bv) - 1) < UNSTABLE_ZERO
        for a in c.existing:
            av = np.array(a)
            assert abs(av.dot(bv)) < UNSTABLE_ZERO
