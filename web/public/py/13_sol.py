def mat_tra(a: mat) -> mat:
    cols, rows = mat_siz(a)
    result: mat = [[0 for _ in range(cols)] for _ in range(rows)]

    for col in range(cols):
        for row in range(rows):
            result[row][col] = a[col][row]
    
    return result