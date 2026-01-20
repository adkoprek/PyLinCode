# ortho

## Task
Create a function named `ortho` that takes a list of orthogonal vectors and a new vector as its arguments and returns the component of the new vector that is orthogonal to all vectors in the list using the Gram-Schmidt process.

## Input
- `vecs: list[vec]` - List of mutually orthogonal vectors
- `new: vec` - New vector to orthogonalize

## Output
- `vec` - Component of `new` orthogonal to all vectors in `vecs`

## Theory
The Gram-Schmidt orthogonalization process removes components of a vector that are parallel to a set of orthogonal vectors. Given a list of mutually orthogonal vectors $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ and a new vector $\mathbf{u}$, the orthogonal component is computed as:

$$
\mathbf{u}_{\perp} = \mathbf{u} - \sum_{i=1}^{k} \frac{\mathbf{v}_i \cdot \mathbf{u}}{\mathbf{v}_i \cdot \mathbf{v}_i} \mathbf{v}_i
$$

**Steps:**
1. Start with a copy of the new vector
2. For each orthogonal vector $\mathbf{v}_i$ in the list:
   - Compute the projection factor $f_i = \frac{\mathbf{v}_i \cdot \mathbf{u}}{\mathbf{v}_i \cdot \mathbf{v}_i}$
   - Subtract the projection: $\mathbf{u} \leftarrow \mathbf{u} - f_i \mathbf{v}_i$
3. Return the resulting orthogonal vector

## Example
$$
\text{vecs} = \left\{\begin{pmatrix} 1 \\ 0 \end{pmatrix}\right\}, \quad \mathbf{u} = \begin{pmatrix} 3 \\ 4 \end{pmatrix} \quad \Rightarrow \quad \mathbf{u}_{\perp} = \begin{pmatrix} 0 \\ 4 \end{pmatrix}
$$

## Test
```python
vecs: list[vec] = [[1, 0]]
new: vec = [3, 4]
result = ortho(vecs, new)
np.testing.assert_allclose(result, [0, 4], atol=1e-10)
```

## Cases
-   Test Cases: $50$
-   Dimension: $1$ to $10$
-   Tolerance: $1e-10$