# solve

## Task
Create a function named `solve` that takes a matrix and a vector as its arguments and solves the linear system $A\mathbf{x} = \mathbf{b}$ using LU decomposition. If $A$ is singular, your function must raise a `SingularError`. If the dimensions of $A$ and $\mathbf{b}$ are incompatible or $A$ is not square, your function must raise a `ShapeMismatchedError`.

You must implement the following helper functions:
- `for_sub(L: mat, b: vec) -> vec` - Performs forward substitution to solve $L\mathbf{y} = \mathbf{b}$
- `bck_sub(U: mat, y: vec) -> vec` - Performs backward substitution to solve $U\mathbf{x} = \mathbf{y}$

## Input
- `A: mat` - Coefficient matrix (must be square)
- `b: vec` - Right-hand side vector

## Output
- `vec` - Solution vector $\mathbf{x}$

## Theory
To solve $A\mathbf{x} = \mathbf{b}$ using LU decomposition:

**Step 1:** Verify that $A$ is square. If not, raise `ShapeMismatchedError`.

**Step 2:** Compute the LU decomposition with partial pivoting: $PA = LU$ using the `lu` function.

**Step 3:** Apply the permutation to the right-hand side: $\mathbf{b}' = P\mathbf{b}$ using `mat_vec_mul`.

**Step 4:** Solve $L\mathbf{y} = \mathbf{b}'$ for $\mathbf{y}$ using forward substitution (`for_sub`).

**Step 5:** Solve $U\mathbf{x} = \mathbf{y}$ for $\mathbf{x}$ using backward substitution (`bck_sub`). If a diagonal element of $U$ is zero (within machine precision), raise `SingularError`.

## Example
$$
\begin{pmatrix}
2 & 1 \\
1 & 3
\end{pmatrix}
\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} =
\begin{pmatrix} 1 \\ 2 \end{pmatrix}
\quad \Rightarrow \quad
\mathbf{x} =
\begin{pmatrix} 0.2 \\ 0.6 \end{pmatrix}
$$

## Test
```python
A: mat = [[2, 1], [1, 3]]
b: vec = [1, 2]
x = solve(A, b)
n = len(A)
m = len(b)
assert x.shape == (n,), f"x must be shape ({n},), got {x.shape}"
assert b.shape == (m,), f"b must be shape ({m},), got {b.shape}"
np.testing.assert_allclose(A @ x, b, atol=1e-14)
```

## Cases
-   Test Cases: $50$ (for `solve`, `for_sub`, and `bck_sub`)
-   Error Test Cases: $5$ (for `solve`)
-   Error Test Cases: $5$ (for `bck_sub` with `SingularError`)
-   Dimension: $1$ to $10$ (rows and columns)
-   Tolerance: $10^{-14}$