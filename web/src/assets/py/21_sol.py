def ortho_base(vecs: list[vec]) -> tuple[vec]:
    result = [copy(vecs[0])]
    
    for i in range(1, len(vecs)):
        result.append(ortho(result, vecs[i]))

    return tuple(result)