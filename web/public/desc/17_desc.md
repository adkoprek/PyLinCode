# inv

## Task
Create a function named `inv` that takes a matrix as its argument and returns its inverse. If $A$ is singular, your function must raise a `SingularError`. If $A$ is not square, your function must raise a `ShapeMismatchedError`.

## Input
- `a: mat` - Input matrix (must be square and non-singular)

## Output
- `mat` - Inverse of the input matrix

## Theory
To compute the matrix inverse $A^{-1}$ using LU decomposition:

**Step 1:** Verify that $A$ is square. If not, raise `ShapeMismatchedError`.

**Step 2:** Compute the LU decomposition with partial pivoting: $PA = LU$ using the `lu` function.

**Step 3:** Create the identity matrix $I$ of size $n \times n$.

**Step 4:** For each column $i$ of the identity matrix:
- Extract column $\mathbf{e}_i$ from $I$
- Apply the permutation: $\mathbf{e}_i' = P\mathbf{e}_i$
- Solve $L\mathbf{y}_i = \mathbf{e}_i'$ using `for_sub`
- Solve $U\mathbf{x}_i = \mathbf{y}_i$ using `bck_sub` (will raise `SingularError` if matrix is singular)
- Column $\mathbf{x}_i$ becomes the $i$-th column of $A^{-1}$

**Step 5:** Transpose the result to obtain $A^{-1}$.

## Example
$$
\begin{pmatrix}
2 & 1 \\
1 & 3
\end{pmatrix}^{-1} =
\begin{pmatrix}
0.6 & -0.2 \\
-0.2 & 0.4
\end{pmatrix}
$$

## Test
```python
A: mat = [[2, 1], [1, 3]]
A_inv = inv(A)
np.testing.assert_allclose(A_inv, [[0.6, -0.2], [-0.2, 0.4]], atol=1e-14)
```

## Cases
-   Test Cases: $50$
-   Error Test Cases: $5$ (for `SingularError`)
-   Dimension: $1$ to $10$ (rows and columns)
-   Tolerance: $1e-14$