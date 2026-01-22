# tests/lesson_18_mat_prj.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
from src.types import mat
from src.errors import SingularError
import src.mat_prj as mat_prj

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
    globals_before = set(mat_prj.__dict__.keys())

    for c in load_cases():
        ca_copy = copy(c.a)

        result = mat_prj.mat_prj(c.a.tolist())
        np.testing.assert_allclose(result, c.result, atol=UNSTABLE_ZERO)

        np.testing.assert_equal(ca_copy, c.a, "You changed the input a")

    globals_after = set(mat_prj.__dict__.keys())
    new_globals = globals_after - globals_before
    assert not new_globals, f"You created a global variable {new_globals} which is forbidden"