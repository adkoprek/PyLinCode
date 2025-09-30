def lu(a: mat) -> tuple[mat, mat, mat]:
    rows, cols = mat_siz(a)

    U = copy(a)
    L = [[0]*cols for _ in range(rows)]
    P = mat_ide(rows)

    for i in range(rows):
        pivot = max(range(i, rows), key=lambda r: abs(U[r][i]))

        if pivot != i:
            _swap_rows(U, i, pivot)
            _swap_rows(P, i, pivot)

            L[i][:i], L[pivot][:i] = L[pivot][:i], L[i][:i]

        for j in range(i+1, rows):
            if U[i][i] == 0:
                continue

            fac = U[j][i] / U[i][i]
            L[j][i] = fac

            for k in range(i, cols):
                U[j][k] -= fac * U[i][k]

        L[i][i] = 1.0

    return L, U, P