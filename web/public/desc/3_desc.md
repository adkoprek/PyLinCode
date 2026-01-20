# vec_dot

## Task
Create a function named `vec_dot` that takes two vectors as its arguments and returns their dot product. If the dimensions of the two vectors differ, your function must raise a `ShapeMismatchedError`.

## Input
- `a: vec` - First input vector
- `b: vec` - Second input vector

## Output
- `float` - Dot product of the two input vectors

## Example
$$
\begin{pmatrix} 2 \\ 3 \\ 4 \end{pmatrix} \cdot \begin{pmatrix} 5 \\ 6 \\ 7 \end{pmatrix} = 2 \cdot5 + 3 \cdot6 + 4 \cdot 7 = 56 
$$

## Test
```python
a: vec = [2, 3, 4]
b: vec = [5, 6, 7]
assert vec_dot(a, b) == 56
```

## Cases
-   Test Cases: $50$
-   Error Test Cases: $5$
-   Dimension: $1$ to $10$
-   Tolerance: $0$