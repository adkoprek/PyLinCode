# tests/lesson_10_mat_ide.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
import src.mat_ide as mat_ide
from src.types import mat


@dataclass
class Case:
    size: int
    result: mat


def load_cases():
    cases = []
    for _ in range(TEST_CASES):
        size = np.random.randint(1, 10)
        cases.append(Case(size, np.eye(size)))
    return cases


def run():
    globals_before = set(mat_ide.__dict__.keys())

    for c in load_cases():
        np.testing.assert_allclose(mat_ide.mat_ide(c.size), c.result, atol=0)

    globals_after = set(mat_ide.__dict__.keys())
    new_globals = globals_after - globals_before
    assert not new_globals, f"You created a global variable {new_globals} which is forbidden"