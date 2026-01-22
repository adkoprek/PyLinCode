# tests/lesson_12_mat_tra.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
import src.mat_tra as mat_tra
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
    globals_before = set(mat_tra.__dict__.keys())

    for c in load_cases():
        cA_copy = copy(c.A)

        np.testing.assert_allclose(mat_tra.mat_tra(c.A.tolist()), c.result, atol=0)

        np.testing.assert_equal(cA_copy, c.A, "You changed the input A")

    globals_after = set(mat_tra.__dict__.keys())
    new_globals = globals_after - globals_before
    assert not new_globals, f"You created a global variable {new_globals} which is forbidden"