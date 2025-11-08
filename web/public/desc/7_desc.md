# mat_add

## Task
Create a function named `mat_add` that takes two matrices as its arguments and returns their element-wise sum. If the dimensions of the two matrices differ, your function must raise a `ShapeMismatchedError`.

## Input
- `A: mat` - First input matrix
- `B: mat` - Second input matrix

## Output
- `mat` - Element-wise sum of the two input matrices

## Example
$$
\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} + \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} = \begin{pmatrix} 6 & 8 \\ 10 & 12 \end{pmatrix}
$$

## Test
```python
A: mat = [[1, 2, 3], [4, 5, 6]]
B: mat = [[6, 5, 4], [3, 2, 1]]
assert mat_add(A, B) == [[7, 7, 7], [7, 7, 7]]
```

## Cases
-   Test Cases: $50$
-   Error Test Cases: $5$
-   Dimension: $1$ to $10$ (rows and columns)
-   Tolerance: $0$