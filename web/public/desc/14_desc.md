# lu

## Task
Create a function named `lu` that takes a matrix as its argument and returns its LU decomposition with partial pivoting as a tuple of three matrices: L (lower triangular), U (upper triangular), and P (permutation matrix), such that PA = LU.

## Input
- `a: mat` - Input matrix to decompose

## Output
- `tuple[mat, mat, mat]` - Tuple containing (L, U, P) matrices

## Theory
LU decomposition with partial pivoting factors a matrix A into the product of three matrices: a permutation matrix P, a lower triangular matrix L with ones on the diagonal, and an upper triangular matrix U, such that PA = LU.

Partial pivoting involves swapping rows at each step to place the largest absolute value element in the pivot position. Here's a step-by-step example:

Starting with:
$$
A = \begin{pmatrix} 2 & 1 \\ 4 & 3 \end{pmatrix}
$$

**Step 1:** Find the largest absolute value in column 1 (positions (1,1) and (2,1)). Since |4| > |2|, swap rows 1 and 2:
$$
\begin{pmatrix} 4 & 3 \\ 2 & 1 \end{pmatrix}, \quad P = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
$$

**Step 2:** Eliminate below the pivot. The multiplier for row 2 is $\frac{2}{4} = 0.5$. Subtract 0.5 times row 1 from row 2:
$$
U = \begin{pmatrix} 4 & 3 \\ 0 & -0.5 \end{pmatrix}, \quad L = \begin{pmatrix} 1 & 0 \\ 0.5 & 1 \end{pmatrix}
$$

The result satisfies PA = LU.

## Example
$$
\begin{pmatrix} 2 & 1 \\ 4 & 3 \end{pmatrix} \rightarrow
L = \begin{pmatrix} 1 & 0 \\ 0.5 & 1 \end{pmatrix}, 
U = \begin{pmatrix} 4 & 3 \\ 0 & -0.5 \end{pmatrix}, 
P = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
$$

## Test
```python
A: mat = [[2, 1], [4, 3]]
L, U, P = lu(A)
n = len(A)
assert L.shape == (n, n), "L must be same shape as A"
assert U.shape == (n, n), "U must be same shape as A"
assert P.shape == (n, n), "P must be same shape as A"
assert np.allclose(P @ P.T, np.eye(n), atol=1e-14), "P is not orthogonal (not permutation)"
assert np.allclose(np.sum(P, axis=0), 1), "Each column of P must have exactly one 1"
assert np.allclose(np.sum(P, axis=1), 1), "Each row of P must have exactly one 1"
assert np.allclose(np.tril(L), L, atol=1e-14), "L must be lower triangular"
assert np.allclose(np.diag(L), np.ones(n), atol=1e-14), "Diagonal of L must be all ones"
assert np.allclose(np.triu(U), U, atol=1e-14), "U must be upper triangular"
assert np.allclose(P @ A, L @ U, atol=1e-14), "Decomposition check failed: P*A != L*U"
```

## Cases
-   Test Cases: $50$
-   Dimension: $1$ to $10$ (rows and columns)
-   Tolerance: $1e-14$