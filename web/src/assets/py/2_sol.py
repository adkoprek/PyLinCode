def vec_scl(a: vec, s: float) -> vec:
    result: vec = []
    for e in a:
        result.append(s * e)
    return result