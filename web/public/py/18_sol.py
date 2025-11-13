def mat_prj(a: mat) -> mat:
    return mat_mul(
                mat_mul(
                    a,
                    inv(
                        mat_mul(
                            mat_tra(a),
                            a
                        )
                    )
                ),
                mat_tra(a)
            )