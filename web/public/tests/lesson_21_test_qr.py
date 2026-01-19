# tests/lesson_20_qr.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
from src.types import mat
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