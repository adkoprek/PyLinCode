# tests/lesson_12_mat_tra.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
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
        cA_copy = copy(c.A)

        np.testing.assert_allclose(mat_tra(c.A), c.result, atol=0)

        np.testing.assert_equal(cA_copy, c.A, "You changed the input A")
