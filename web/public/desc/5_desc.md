# vec_nor

## Task
Create a function named `vec_nor` that takes a vector as its argument and returns the normalized vector (unit vector in the same direction).

## Input
- `a: vec` - Input vector

## Output
- `vec` - Normalized vector with length 1

## Example
$$
\text{normalize}\left(\begin{pmatrix} 3 \\ 4 \end{pmatrix}\right) = \frac{1}{5}\begin{pmatrix} 3 \\ 4 \end{pmatrix} = \begin{pmatrix} 0.6 \\ 0.8 \end{pmatrix}
$$

## Test
```python
a: vec = [3, 4]
result = vec_nor(a)
np.testing.assert_allclose(result, [0.6, 0.8])
```

## Cases
-   Test Cases: $50$
-   Dimension: $1$ to $10$
-   Tolerance: $0$