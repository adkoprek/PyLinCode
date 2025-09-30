def mat_add(a: mat, b: mat) -> mat:
    size_a = mat_siz(a)
    size_b = mat_siz(b)

    if size_a != size_b:
        raise ShapeMismatchedError(f"The size of the matrix a ({size_a}) does not match the size of b ({size_b})")

    result = []
    for row_a, row_b in zip(a, b):
        result.append([])
        for e_a, e_b in zip(row_a, row_b):
            result[-1].append(e_a + e_b)

    return result