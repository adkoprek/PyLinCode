# tests/lesson_2_vec_scl.py
import random
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
import src.vec_scl as vec_scl
from src.types import vec


@dataclass
class Case:
    a: vec
    s: float
    result: vec


def load_cases():
    return [
        Case(a := random_vector(), s := random.randint(1, 100), s * a)
        for _ in range(TEST_CASES)
    ]


def run():
    globals_before = set(vec_scl.__dict__.keys())

    for c in load_cases():
        ca_copy = copy(c.a)

        r = vec_scl.vec_scl(c.a.tolist(), c.s)
        np.testing.assert_allclose(r, c.result, atol=0)

        np.testing.assert_equal(ca_copy, c.a, "You changed the input a")

    globals_after = set(vec_scl.__dict__.keys())
    new_globals = globals_after - globals_before
    assert not new_globals, f"You created a global variable {new_globals} which is forbidden"
