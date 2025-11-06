def vec_add(a: vec, b: vec) -> vec:
    if len(a) != len(b):
        raise ShapeMismatchedError(f"The size of the vector a ({len(a)}) dosent match vector b ({len(b)})")

    result: vec = []
    for e_a, e_b in zip(a, b):
        result.append(e_a + e_b)

    return result