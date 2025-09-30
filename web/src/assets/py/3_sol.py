def vec_dot(a: vec, b: vec) -> float:
    if len(a) != len(b):
        raise ShapeMismatchedError(f"The size of the vector a ({len(a)}) dosent match vector b ({len(b)})")

    result: float = 0
    for e_a, e_b in zip(a, b):
        result += e_a * e_b

    return result