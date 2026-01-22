# tests/lesson_13_mat_vec_mul.py
import numpy as np
from dataclasses import dataclass
from copy import copy

from tests.consts import *
from src.errors import ShapeMismatchedError
import src.mat_vec_mul as mat_vec_mul
from src.types import mat, vec


@dataclass
class Case:
    a: mat
    v: vec
    result: vec | Exception
    error: bool = False


def load_cases():
    cases = []

    for _ in range(TEST_CASES):
        A = random_matrix()
        x = random_vector(A.shape[1])
        cases.append(Case(A, x, np.matmul(A, x)))

    for _ in range(ERROR_TEST_CASES):
        A = random_matrix()
        x = random_vector(A.shape[1] + np.random.randint(1, 10))
        cases.append(Case(A, x, ShapeMismatchedError, error=True))

    return cases


def run():
    globals_before = set(mat_vec_mul.__dict__.keys())

    for c in load_cases():
        if c.error:
            try:
                mat_vec_mul.mat_vec_mul(c.a, c.v)
            except c.result:
                continue
            raise AssertionError("mat_vec_mul: expected ShapeMismatchedError")
        else:
            cA_copy = copy(c.a)
            cv_copy = copy(c.v)

            np.testing.assert_allclose(mat_vec_mul.mat_vec_mul(c.a.tolist(), c.v.tolist()), c.result, atol=0)

            np.testing.assert_equal(cA_copy, c.a, "You changed the input A")
            np.testing.assert_equal(cv_copy, c.v, "You changed the input v")

    globals_after = set(mat_vec_mul.__dict__.keys())
    new_globals = globals_after - globals_before
    assert not new_globals, f"You created a global variable {new_globals} which is forbidden"