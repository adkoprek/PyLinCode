# tests/lesson_3_vec_dot.py
import random
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
from src.vec_scl import vec_scl
import src.vec_dot as vec_dot
from src.errors import ShapeMismatchedError
from src.types import vec


@dataclass
class Case:
    a: vec
    b: vec
    result: float | Exception
    error: bool = False


def load_cases():
    cases = []

    for _ in range(TEST_CASES):
        a = random_vector()
        b = random_vector(a.shape)
        cases.append(Case(a, b, np.dot(a, b)))

    for _ in range(ERROR_TEST_CASES):
        a = random_vector()
        b = random_vector((a.shape[0] + random.randint(1, 10),))
        cases.append(Case(a, b, ShapeMismatchedError, error=True))

    return cases


def run():
    globals_before = set(vec_dot.__dict__.keys())

    for c in load_cases():
        if c.error:
            try:
                vec_dot.vec_dot(c.a, c.b)
            except c.result:
                continue
            raise AssertionError("vec_dot: expected ShapeMismatchedError")
        else:
            ca_copy = copy(c.a)
            cb_copy = copy(c.b)

            r = vec_dot.vec_dot(c.a.tolist(), c.b.tolist())
            np.testing.assert_allclose(r, c.result, atol=0)
            np.testing.assert_allclose(vec_dot.vec_dot(vec_scl(c.a, 2), c.b), 2 * r)

            np.testing.assert_equal(ca_copy, c.a, "You changed the input a")
            np.testing.assert_equal(cb_copy, c.b, "You changed the input b")

    globals_after = set(vec_dot.__dict__.keys())
    new_globals = globals_after - globals_before
    assert not new_globals, f"You created a global variable {new_globals} which is forbidden"