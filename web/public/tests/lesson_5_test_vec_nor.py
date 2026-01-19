# tests/lesson_5_vec_nor.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
from src.vec_nor import vec_nor
from src.vec_len import vec_len
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
    for c in load_cases():
        ca_copy = copy(c.a)

        r = vec_nor(c.a.tolist())
        np.testing.assert_allclose(r, c.result, atol=1e-10)
        np.testing.assert_allclose(vec_len(r), 1.0, atol=1e-10)

        np.testing.assert_equal(ca_copy, c.a, "You changed the input a")
