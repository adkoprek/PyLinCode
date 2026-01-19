# mat_prj

## Task
Create a function named `mat_prj` that takes a matrix as its argument and returns the projection matrix onto the column space of $A$.

## Input
- `a: mat` - Input matrix

## Output
- `mat` - Projection matrix onto the column space of $A$

## Theory
The projection matrix $P$ that projects vectors onto the column space of matrix $A$ is given by:
$$
P = A(A^TA)^{-1}A^T
$$

This matrix has the property that for any vector $\mathbf{b}$, the product $P\mathbf{b}$ gives the projection of $\mathbf{b}$ onto the column space of $A$.


## Example
$$
A = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad P = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}
$$

## Test
```python
a: mat = [[1], [0]]
result = mat_prj(a)
np.testing.assert_allclose(result, [[1, 0], [0, 0]], atol=1e-10)
```

## Cases
-   Test Cases: $50$
-   Dimension: $1$ to $10$ (rows and columns)
-   Tolerance: $1e-10$