def mat_col(a: mat, index: int) -> vec:
    cols, _ = mat_siz(a)

    col: vec = []
    for i in range(cols):
        col.append(a[i][index])

    return col