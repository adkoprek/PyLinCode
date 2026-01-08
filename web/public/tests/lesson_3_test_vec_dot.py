# tests/lesson_3_vec_dot.py
import random
import numpy as np
from dataclasses import dataclass

from tests.consts import *
from src.vec_add import vec_add
from src.vec_dot import vec_dot
from src.vec_scl import vec_scl
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
    for c in load_cases():
        if c.error:
            try:
                vec_dot(c.a, c.b)
            except c.result:
                continue
            raise AssertionError("vec_dot: expected ShapeMismatchedError")
        else:
            r = vec_dot(c.a, c.b)
            np.testing.assert_allclose(r, c.result, atol=0)
            np.testing.assert_allclose(vec_dot(vec_scl(c.a, 2), c.b), 2 * r)
