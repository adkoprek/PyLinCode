# tests/lesson_17_vec_prj.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
from src.types import vec
import src.vec_prj as vec_prj

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
    globals_before = set(vec_prj.__dict__.keys())

    for c in load_cases():
        ca_copy = copy(c.a)
        cb_copy = copy(c.b)

        result = vec_prj.vec_prj(c.a.tolist(), c.b.tolist())
        np.testing.assert_allclose(result, c.result, atol=0)

        np.testing.assert_equal(ca_copy, c.a, "You changed the input a")
        np.testing.assert_equal(cb_copy, c.b, "You changed the input b")

    globals_after = set(vec_prj.__dict__.keys())
    new_globals = globals_after - globals_before
    assert not new_globals, f"You created a global variable {new_globals} which is forbidden"