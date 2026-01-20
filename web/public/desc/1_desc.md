# vec_add

## Task
Create a function named `vec_add` that takes two vectors as its arguments and returns their sum. If the dimensions of the two vectors differ, your function must raise a `ShapeMismatchedError`.

## Input
- `a: vec` - First input vector
- `b: vec` - Second input vector

## Output
- `vec` - Sum of the two input vectors

## Example
$$
\begin{pmatrix} 1 \\ 10 \\ 2 \\ 6 \end{pmatrix} + \begin{pmatrix} 8 \\ 12 \\ 8 \\ 1 \end{pmatrix} = \begin{pmatrix} 9 \\ 22 \\ 10 \\ 7 \end{pmatrix}
$$

## Test
```python
a: vec = [1, 10, 2, 6]
b: vec = [8, 12, 8, 1]
assert vec_add(a, b) == [9, 22, 10, 7]
```

## Cases
-   Test Cases: $50$
-   Error Test Cases: $5$
-   Dimension: $1$ to $10$
-   Tolerance: $0$