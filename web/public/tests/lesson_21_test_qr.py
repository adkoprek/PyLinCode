# tests/lesson_20_qr.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
from src.types import mat
import src.qr as qr

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
    globals_before = set(qr.__dict__.keys())

    for c in load_cases():
        ca_copy = copy(c.a)

        Q, R = qr.qr(c.a.tolist())
        QM = np.array(Q)
        RM = np.array(R)
        np.testing.assert_allclose(QM @ QM.T, np.eye(len(Q)), atol=UNSTABLE_ZERO)
        np.testing.assert_allclose(QM @ RM, c.a, atol=UNSTABLE_ZERO)
        m, n = np.array(c.a).shape
        assert QM.shape == (m, m)
        assert RM.shape == (m, n)
        assert np.allclose(RM, np.triu(RM), atol=UNSTABLE_ZERO)

        np.testing.assert_equal(ca_copy, c.a, "Vou changed the input a")

    globals_after = set(qr.__dict__.keys())
    new_globals = globals_after - globals_before
    assert not new_globals, f"You created a global variable {new_globals} which is forbidden"