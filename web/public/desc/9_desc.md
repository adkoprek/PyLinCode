# mat_row

## Task
Create a function named `mat_row` that takes a matrix and an index as its arguments and returns the specified row of the matrix (zero-indexed).

## Input
- `A: mat` - Input matrix
- `i: int` - Row index (zero-indexed)

## Output
- `vec` - The $i$-th row of the matrix

## Example
$$
\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} 
\rightarrow
\begin{pmatrix} 1 & 2 \end{pmatrix} 
\text{ (0th row)}
$$

## Test
```python
A: mat = [[1, 2, 3], [4, 5, 6]]
assert mat_row(A, 1) == [4, 5, 6]
```

## Cases
-   Test Cases: $50$
-   Dimension: $1$ to $10$ (rows and columns)
-   Tolerance: $0$