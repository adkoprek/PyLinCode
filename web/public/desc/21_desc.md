# det

## Task
Create a function named `det` that takes a matrix as its argument and returns its determinant using LU decomposition. If the matrix is not square, your function must raise a `ShapeMismatchedError`.

## Input
- `a: mat` - Input matrix (must be square)

## Output
- `float` - Determinant of the input matrix

## Theory
The determinant can be efficiently computed using LU decomposition with partial pivoting. Given $PA = LU$, the determinant is:

$$
\det(A) = \det(P^{-1}LU) = \det(P^{-1}) \cdot \det(L) \cdot \det(U)
$$

Since:
- $\det(L) = 1$ (L has ones on the diagonal)
- $\det(U) = \prod_{i=1}^{n} U_{ii}$ (product of diagonal elements)
- $\det(P^{-1}) = \det(P^T) = \det(P) = (-1)^{\text{number of swaps}}$

**Steps:**

1. Verify that $A$ is square. If not, raise `ShapeMismatchedError`.
2. Compute the LU decomposition: $(L, U, P) = \text{lu}(A)$
3. Compute $\det(U)$ as the product of diagonal elements: $\prod_{i=0}^{n-1} U_{ii}$
4. Determine the sign from the permutation matrix $P$:

**Computing the Sign of P:**

The sign of a permutation equals $(-1)^{\text{number of transpositions}}$. We can compute this using cycle decomposition:

- Convert the permutation matrix to a permutation array: for each row $i$, find which column contains the 1, giving $p[i]$
- Count the number of disjoint cycles in the permutation:
  - Start with all positions unvisited
  - For each unvisited position $i$:
    * Increment cycle count
    * Follow the chain: $i \to p[i] \to p[p[i]] \to \ldots$ until returning to $i$
    * Mark all positions in this cycle as visited
- The number of transpositions = $n - \text{number of cycles}$
- Sign = $(-1)^{n - \text{cycles}}$

**Example:** For $P = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{pmatrix}$, the permutation is $p = [1, 2, 0]$. This forms one cycle: $0 \to 1 \to 2 \to 0$. So cycles = 1, and sign = $(-1)^{3-1} = 1$.

5. Return: $\det(A) = \text{sign} \times \det(U)$

## Example
$$
\det\begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix} = 6 - 1 = 5
$$

## Test
```python
a: mat = [[2, 1], [1, 3]]
result = det(a)
np.testing.assert_allclose(result, 5, atol=1e-14)
```

## Cases
-   Test Cases: $50$
-   Error Test Cases: $5$
-   Dimension: $1$ to $10$ (rows and columns)
-   Tolerance: $1e-14$