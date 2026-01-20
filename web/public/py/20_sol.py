def ortho(vecs: list[vec], new: vec) -> vec:
    result = copy(new)

    for o_vec in vecs:
        f = vec_dot(o_vec, result) / vec_dot(o_vec, o_vec)
        result = vec_add(result, vec_scl(o_vec, -f))

    return vec_nor(result)