# tests/lesson_1_vec_add.py
import random
import numpy as np
from dataclasses import dataclass

from tests.consts import *
from src.vec_add import vec_add
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
    for c in load_cases():
        if c.error:
            try:
                vec_add(c.a, c.b)
            except c.result:
                continue
            raise AssertionError("vec_add: expected ShapeMismatchedError")
        else:
            np.testing.assert_allclose(vec_add(c.a, c.b), c.result, atol=0)
