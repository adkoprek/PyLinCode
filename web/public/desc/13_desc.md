# mat_tra

## Task
Create a function named `mat_tra` that takes a matrix as its argument and returns its transpose.

## Input
- `a: mat` - Input matrix

## Output
- `mat` - Transpose of the input matrix

## Example
$$
\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}^T =
\begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}
$$

## Test
```python
a: mat = [[1, 2, 3], [4, 5, 6]]
result = mat_tra(a)
np.testing.assert_allclose(result, [[1, 4], [2, 5], [3, 6]])
```

## Cases
-   Test Cases: $50$
-   Dimension: $1$ to $10$ (rows and columns)
-   Tolerance: $0$