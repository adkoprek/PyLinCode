# Tolerances for unstable algorithems
ZERO = 1e-14
UNSTABLE_ZERO = 1e-10

# Number of tests to run and number of errors checks per funtion
TEST_CASES = 50
ERROR_TEST_CASES = 5

# Internal types
mat = list[list[float]]
vec = list[list[float]]

# Functions for testing to create random matricies and vectors
import numpy as np
def random_matrix(shape: tuple[int, int] = None, square: bool = False) -> mat:
    if shape == None:
        rows = randint(2, 10)
        cols = randint(2, 10)
        if square:
            rows = cols
        return np.random.random((rows, cols))

    return np.random.random(shape)

def random_vector(shape: int = None) -> vec:
    if shape == None:
        length = randint(2, 10)
        return np.random.random(length)

    return np.random.random(shape)

# Colors for the terminal
class colors:
    HEADER =    '\033[95m'
    OKBLUE =    '\033[94m'
    OKCYAN =    '\033[96m'
    OKGREEN =   '\033[92m'
    WARNING =   '\033[93m'
    FAIL =      '\033[91m'
    ENDC =      '\033[0m'
    BOLD =      '\033[1m'
    UNDERLINE = '\033[4m'

# Internal logging functions
def log_error(message: str):
    print(f"{colors.FAIL} [LinAlg]: {message} {colors.ENDC}")

def log_info(message: str):
    print(f"[LinAlg]: {message}")

# Error two throw when Size of operandes do not match
class ShapeMismatchedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        log_error(message)

# Error two throw in solving of linear systems
class SingularError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        log_error(message)
