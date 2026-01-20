# tests/lesson_19_ortho.py
import numpy as np
from dataclasses import dataclass
from random import randint
from copy import copy

from tests.consts import *
from src.types import vec
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
        cexisting_copy = copy(c.existing)
        cnew_copy = copy(c.new)

        orthogonalized = ortho(c.existing, c.new)
        bv = np.array(orthogonalized)
        assert abs(np.linalg.norm(bv) - 1) < UNSTABLE_ZERO
        for a in c.existing:
            av = np.array(a)
            assert abs(av.dot(bv)) < UNSTABLE_ZERO

        np.testing.assert_equal(cexisting_copy, c.existing, "You changed the input exisiting")
        np.testing.assert_equal(cnew_copy, c.new, "You changed the input new")
