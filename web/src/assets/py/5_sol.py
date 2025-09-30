def vec_nor(a: vec) -> vec:
    length = vec_len(a)

    result: vec = []
    for e in a:
        result.append(e / length)

    return result

