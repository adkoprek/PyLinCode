def mat_scl(a: mat, s: int) -> mat:
    result: vec = []
    for row in a:
        result.append([])
        for e in row:
            result[-1].append(e * s)

    return result