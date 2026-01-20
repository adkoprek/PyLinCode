# mat_col

## Task
Create a function named `mat_col` that takes a matrix and an index as its arguments and returns the specified column of the matrix (zero-indexed).

## Input
- `A: mat` - Input matrix
- `j: int` - Column index (zero-indexed)

## Output
- `vec` - The $j$-th column of the matrix

## Example
$$
\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} 
\rightarrow
\begin{pmatrix} 2 \\ 4 \end{pmatrix} 
\text{ (1st column)}
$$

## Test
```python
A: mat = [[1, 2, 3], [4, 5, 6]]
assert mat_col(A, 1) == [2, 5]
```

## Cases
-   Test Cases: $50$
-   Dimension: $1$ to $10$ (rows and columns)
-   Tolerance: $0$