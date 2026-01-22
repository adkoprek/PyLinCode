# tests/lesson_5_vec_nor.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
import src.vec_nor as vec_nor
import src.vec_len as vec_len
from src.types import vec


@dataclass
class Case:
    a: vec
    result: vec


def load_cases():
    return [
        Case(a := random_vector(), a / np.linalg.norm(a))
        for _ in range(TEST_CASES)
    ]


def run():
    globals_before = set(vec_nor.__dict__.keys())

    for c in load_cases():
        ca_copy = copy(c.a)

        r = vec_nor.vec_nor(c.a.tolist())
        np.testing.assert_allclose(r, c.result, atol=1e-10)
        np.testing.assert_allclose(vec_len.vec_len(r), 1.0, atol=1e-10)

        np.testing.assert_equal(ca_copy, c.a, "You changed the input a")

    globals_after = set(vec_nor.__dict__.keys())
    new_globals = globals_after - globals_before
    assert not new_globals, f"You created a global variable {new_globals} which is forbidden"