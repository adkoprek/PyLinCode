# tests/lesson_2_vec_scl.py
import random
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
from src.vec_scl import vec_scl
from src.vec_add import vec_add
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
    for c in load_cases():
        ca_copy = copy(c.a)

        r = vec_scl(c.a, c.s)
        np.testing.assert_allclose(r, c.result, atol=0)

        np.testing.assert_equal(ca_copy, c.a, "You changed the input a")

