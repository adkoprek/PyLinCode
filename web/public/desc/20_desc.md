# qr

## Task
Create a function named `qr` that takes a matrix as its argument and returns its QR decomposition using the Householder reflection method. The function returns a tuple of two matrices: Q (orthogonal matrix) and R (upper triangular matrix), such that $A = QR$.

## Input
- `a: mat` - Input matrix

## Output
- `tuple[mat, mat]` - Tuple containing (Q, R) matrices

## Theory
The QR decomposition factors a matrix $A$ into the product of an orthogonal matrix $Q$ and an upper triangular matrix $R$, such that $A = QR$. The Householder reflection method achieves this by applying a sequence of orthogonal transformations to progressively zero out elements below the diagonal.

**Householder Reflection Steps:**

For each column $i$ (from 0 to $\min(m-1, n)$):

1. Extract the subvector $\mathbf{x}$ from column $i$ starting at row $i$
2. Compute its length: $\|\mathbf{x}\|$
3. Determine the sign for numerical stability and compute:
   $$\alpha = -\text{sign}(x_0) \cdot \|\mathbf{x}\|$$
4. Create the reflection vector:
   $$\mathbf{u} = \mathbf{x} - \alpha \mathbf{e}_1$$
   where $\mathbf{e}_1 = (1, 0, 0, \ldots)$
5. Normalize: $\mathbf{v} = \frac{\mathbf{u}}{\|\mathbf{u}\|}$
6. Construct the Householder matrix:
   $$H = I - 2\mathbf{v}\mathbf{v}^T$$
7. Update: $R \leftarrow HR$ and $Q \leftarrow QH^T$

The resulting $Q$ is orthogonal ($Q^TQ = I$) and $R$ is upper triangular.

## Example
$$
\begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix} \rightarrow
Q = \begin{pmatrix} 0.707 & 0.707 \\ 0.707 & -0.707 \end{pmatrix}, 
R = \begin{pmatrix} 1.414 & 0.707 \\ 0 & 0.707 \end{pmatrix}
$$

## Test
```python
a: mat = [[1, 1], [1, 0]]
Q, R = qr(a)
m, n = len(a), len(a[0])
assert Q.shape == (m, m), f"Q must be shape ({m}, {m})"
assert R.shape == (m, n), f"R must be shape ({m}, {n})"
np.testing.assert_allclose(Q @ Q.T, np.eye(m), atol=1e-10)
np.testing.assert_allclose(Q @ R, a, atol=1e-10)
assert np.allclose(R, np.triu(R)), "R is not upper triangular"
```

## Cases
-   Test Cases: $50$
-   Dimension: $1$ to $10$ (rows and columns)
-   Tolerance: $1e-10$