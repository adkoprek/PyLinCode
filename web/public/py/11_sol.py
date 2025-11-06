def mat_mul(a: mat, b: mat) -> mat:
    size_a = mat_siz(a) 
    size_b = mat_siz(b)

    if size_a[1] != size_b[0]:
        raise ShapeMismatchedError(
            f"The number of columns of a ({size_a[1]}) does not match the number of rows of b ({size_b[0]})"
        )

    r_rows = size_a[0]
    r_cols = size_b[1]
    result: mat = [[0 for _ in range(r_cols)] for _ in range(r_rows)]


    for i in range(r_cols):
        for j in range(r_rows):
            row = mat_row(a, j)
            col = mat_col(b, i)
            result[j][i] = vec_dot(row, col)

    return result