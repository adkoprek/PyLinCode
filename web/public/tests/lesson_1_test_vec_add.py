# tests/lesson_1_vec_add.py
import random
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
import src.vec_add as vec_add
from src.errors import ShapeMismatchedError
from src.types import vec


@dataclass
class Case:
    a: vec
    b: vec
    result: vec | Exception
    error: bool = False


def load_cases():
    cases = []

    for _ in range(TEST_CASES):
        a = random_vector()
        b = random_vector(a.shape)
        cases.append(Case(a, b, a + b))

    for _ in range(ERROR_TEST_CASES):
        a = random_vector()
        b = random_vector((a.shape[0] + random.randint(1, 10),))
        cases.append(Case(a, b, ShapeMismatchedError, error=True))

    return cases


def run():
    globals_before = set(vec_add.__dict__.keys())

    for c in load_cases():
        if c.error:
            try:
                vec_add.vec_add(c.a, c.b)
            except c.result:
                continue
            raise AssertionError("vec_add: expected ShapeMismatchedError")
        else:
            ca_copy = copy(c.a)
            cb_copy = copy(c.b)

            np.testing.assert_allclose(vec_add.vec_add(c.a.tolist(), c.b.tolist()), c.result, atol=0)

            np.testing.assert_equal(ca_copy, c.a, "You changed the input a")
            np.testing.assert_equal(cb_copy, c.b, "You changed the input b")

    globals_after = set(vec_add.__dict__.keys())
    new_globals = globals_after - globals_before
    assert not new_globals, f"You created a global variable {new_globals} which is forbidden"
