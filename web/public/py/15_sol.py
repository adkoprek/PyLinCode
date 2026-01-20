def lu(a: mat) -> tuple[mat, mat, mat]:
    rows, cols = mat_siz(a)

    U = copy(a)
    L = [[0]*cols for _ in range(rows)]
    P = mat_ide(rows)
    
    _, colsU = mat_siz(U)
    _, colsP = mat_siz(P)

    for i in range(rows):
        pivot = max(range(i, rows), key=lambda r: abs(U[r][i]))

        if pivot != i:
            for k in range(colsU):
                temp = U[i][k]
                U[i][k] = U[pivot][k]
                U[pivot][k] = temp

            for k in range(colsP):
                temp = P[i][k]
                P[i][k] = P[pivot][k]
                P[pivot][k] = temp

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