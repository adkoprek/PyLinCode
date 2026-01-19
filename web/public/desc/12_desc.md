# mat_mul

## Task
Create a function named `mat_mul` that takes two matrices as its arguments and performs matrix multiplication. If the inner dimensions do not match, your function must raise a `ShapeMismatchedError`.

## Input
- `A: mat` - First input matrix
- `B: mat` - Second input matrix

## Output
- `mat` - Product of the two matrices

## Example
$$
\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}
\begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} =
\begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}
$$

## Test
```python
A: mat = [[1, 2], [3, 4]]
B: mat = [[5, 6], [7, 8]]
assert mat_mul(A, B) == [[19, 22], [43, 50]]
```

## Cases
-   Test Cases: $50$
-   Error Test Cases: $5$
-   Dimension: $1$ to $10$ (rows and columns)
-   Tolerance: $0$