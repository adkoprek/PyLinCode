# mat_vec_mul

## Task
Create a function named `mat_vec_mul` that takes a matrix and a vector as its arguments and returns their product. If the dimensions do not match, your function must raise a `ShapeMismatchedError`.

## Input
- `A: mat` - Input matrix
- `b: vec` - Input vector

## Output
- `vec` - Product of the matrix and vector

## Example
$$
\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}
\begin{pmatrix} 5 \\ 6 \end{pmatrix} =
\begin{pmatrix} 17 \\ 39 \end{pmatrix}
$$

## Test
```python
A: mat = [[1, 2], [3, 4]]
b: vec = [5, 6]
assert mat_vec_mul(A, b) == [17, 39]
```

## Cases
-   Test Cases: $50$
-   Error Test Cases: $5$
-   Dimension: $1$ to $10$ (rows and columns)
-   Tolerance: $0$