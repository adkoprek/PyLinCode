def mat_ide(size: int) -> mat:
    result: mat = [[0 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        result[i][i] = 1

    return result