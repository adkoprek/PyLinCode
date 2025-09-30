def ortho(vecs: list[vec], new: vec, show_factors: bool = False) -> vec | tuple[vec, list[float]]:
    result = copy(new)
    factors: list[float] = []

    for o_vec in vecs:
        f = vec_dot(o_vec, result) / vec_dot(o_vec, o_vec)
        factors.append(f)
        result = vec_add(result, vec_scl(o_vec, -f))

    if show_factors:
        return (result, factors)
    else:
        return result