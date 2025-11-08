# vec_scl

## Task
Create a function named `vec_scl` that takes a scalar value and a vector as its arguments and returns the scalar multiple of the vector.

## Input
- `scalar: float` - Scalar value to multiply by
- `a: vec` - Input vector

## Output
- `vec` - Scalar multiple of the input vector

## Example
$$
3 \cdot \begin{pmatrix} 2 \\ -1 \\ 4 \end{pmatrix}= \begin{pmatrix} 6 \\ -3 \\ 12 \end{pmatrix}
$$

## Test
```python
a: vec = [2, -1, 4]
assert vec_scl(3, a) == [6, -3, 12]
```

## Cases
-   Test Cases: $50$
-   Dimension: $1$ to $10$
-   Tolerance: $0$