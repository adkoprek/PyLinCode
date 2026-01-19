def vec_prj(a: vec, b: vec) -> vec:
    f = vec_dot(a, b) / vec_dot(a, a)
    return vec_scl(a, f)