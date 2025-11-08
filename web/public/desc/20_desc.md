# ortho_base

## Task
Create a function named `ortho_base` that takes a list of linearly independent vectors as its argument and returns a tuple of mutually orthogonal vectors that span the same space using the Gram-Schmidt process.

## Input
- `vecs: list[vec]` - List of linearly independent vectors

## Output
- `tuple[vec]` - Tuple of mutually orthogonal vectors

## Theory
The Gram-Schmidt process converts a set of linearly independent vectors into a set of mutually orthogonal vectors that span the same space.

**Steps:**
1. Keep the first vector unchanged: $\mathbf{u}_1 = \mathbf{v}_1$
2. For each subsequent vector $\mathbf{v}_i$ (where $i = 2, 3, \ldots, n$):
   - Use the `ortho` function to remove all components parallel to the previously computed orthogonal vectors $\{\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_{i-1}\}$
   - The result is $\mathbf{u}_i$, which is orthogonal to all previous vectors
3. Return the tuple of orthogonal vectors $(\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_n)$

## Example
$$
\text{Input: } \left\{\begin{pmatrix} 1 \\ 0 \end{pmatrix}, \begin{pmatrix} 1 \\ 1 \end{pmatrix}\right\} \quad \Rightarrow \quad \text{Output: } \left(\begin{pmatrix} 1 \\ 0 \end{pmatrix}, \begin{pmatrix} 0 \\ 1 \end{pmatrix}\right)
$$

## Test
```python
vecs: list[vec] = [[1, 0], [1, 1]]
result = ortho_base(vecs)
np.testing.assert_allclose(result[0], [1, 0], atol=1e-10)
np.testing.assert_allclose(result[1], [0, 1], atol=1e-10)
```

## Cases
-   Test Cases: $50$
-   Dimension: $1$ to $10$
-   Tolerance: $1e-10$