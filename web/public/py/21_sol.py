def qr(a: mat) -> tuple[mat, mat]:
    rows, cols = mat_siz(a)
    R = copy(a)
    Q = mat_ide(rows)

    for i in range(min(rows - 1, cols)):
        x = copy([row[i] for row in R[i:]])

        length = vec_len(x)
        if length == 0:
            continue

        if x[0] == 0:
            alpha = -length

        elif x[0] < 1:
            alpha = length

        else:
            alpha = -length

        e1 = [0 for _ in range(len(x))]
        e1[0] = 1

        u = vec_add(x, vec_scl(e1, -alpha))

        length = vec_len(u)
        if length < 10e-10:
            continue

        v = vec_nor(u)

        H_sub = mat_add(mat_ide(len(v)), mat_scl(mat_mul(mat_tra([v]), [v]), - 2))
        H_full = mat_ide(rows)
        for row_idx, row in enumerate(H_sub):
            for col_idx, val in enumerate(row):
                H_full[i + row_idx][i + col_idx] = val

        R = mat_mul(H_full, R)
        Q = mat_mul(Q, mat_tra(H_full))

    return (Q, R)
