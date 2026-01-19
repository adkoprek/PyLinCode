def solve(a: mat, b: vec) -> vec:
    rows, cols = mat_siz(a)
    if rows != cols:
        raise ShapeMismatchedError(f"The number of cols ({cols}) and the number of rows ({rows})")

    L, U, P = lu(a)
    bp = mat_vec_mul(P, b)
    y = for_sub(L, bp)
    x = bck_sub(U, y)
    return x

def for_sub(l: mat, b: vec) -> vec:
    rows, _ = mat_siz(l) 
    y = []

    for i in range(rows):
        temp = b[i]
        for col in range(0, i):
            temp -= y[col] * l[i][col]
        y.append(temp / l[i][i])

    return y
        
def bck_sub(u: mat, y: vec) -> vec:
    rows, cols = mat_siz(u) 
    x = []

    for i in range(rows - 1, -1, -1):
        if abs(u[i][i]) < 10e-14:
            raise SingularError("The matrix is singular to machine precision")

        temp = y[i]
        for col in range(rows - 1, i, -1):
            temp -= x[cols - col - 1] * u[i][col]

        x.append(temp / u[i][i])

    x.reverse()
    return x
