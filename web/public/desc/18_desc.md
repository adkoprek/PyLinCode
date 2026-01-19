# vec_prj

## Task
Create a function named `vec_prj` that takes two vectors as its arguments and returns the projection of vector $\mathbf{b}$ onto vector $\mathbf{a}$.

## Input
- `a: vec` - Vector to project onto
- `b: vec` - Vector to be projected

## Output
- `vec` - Projection of $\mathbf{b}$ onto $\mathbf{a}$

## Theory
The projection of vector $\mathbf{b}$ onto vector $\mathbf{a}$ is given by:
$$
\text{proj}_{\mathbf{a}}(\mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{a} \cdot \mathbf{a}} \mathbf{a}
$$

This formula computes the scalar factor $\frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{a} \cdot \mathbf{a}}$ and multiplies it by vector $\mathbf{a}$ to obtain the projection.

## Example
$$
\text{proj}_{\begin{pmatrix} 1 \\ 0 \end{pmatrix}}\begin{pmatrix} 3 \\ 4 \end{pmatrix} = \frac{3}{1} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 3 \\ 0 \end{pmatrix}
$$

## Test
```python
a: vec = [1, 0]
b: vec = [3, 4]
result = vec_prj(a, b)
np.testing.assert_allclose(result, [3, 0])
```

## Cases
-   Test Cases: $50$
-   Dimension: $1$ to $10$
-   Tolerance: $0$