# mat_scl

## Task
Create a function named `mat_scl` that takes a matrix and a scalar as its arguments and returns a new matrix where each element of the matrix is multiplied by the scalar.

## Input
- `A: mat` – Input matrix
- `S: int` – Scalar multiplier

## Output
- `mat` – Matrix resulting from multiplying every element of `A` by `S`

## Example
$$
3 \cdot 
\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}=
\begin{pmatrix}
3 & 6 \\
9 & 12
\end{pmatrix}
$$

## Test
```python
A: mat = [[1, 2, 3], [4, 5, 6]]
S: int = 2
assert mat_scl(A, S) == [[2, 4, 6], [8, 10, 12]]
```

## Cases

  - Test Cases: $50$

  - Error Test Cases: $0$

  - Dimension: $1$ to $10$ (rows and columns)

  - Scalar Range: $-10$ to $10$
