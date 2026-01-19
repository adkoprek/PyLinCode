# tests/lesson_18_mat_prj.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
from src.types import mat
from src.errors import SingularError
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
        ca_copy = copy(c.a)

        result = mat_prj(c.a.tolist())
        np.testing.assert_allclose(result, c.result, atol=UNSTABLE_ZERO)

        np.testing.assert_equal(ca_copy, c.a, "You changed the input a")
