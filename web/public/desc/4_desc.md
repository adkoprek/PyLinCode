# vec_len

## Task
Create a function named `vec_len` that takes a vector as its argument and returns its Euclidean length.

## Input
- `a: vec` - Input vector

## Output
- `float` - Euclidean length of the input vector

## Example
$$
\left\| \begin{pmatrix} 3 \\ 4 \end{pmatrix} \right\| = \sqrt{3^2 + 4^2} = 5
$$

## Test
```python
a: vec = [3, 4]
assert vec_len(a) == 5
```

## Cases
-   Test Cases: $49$
-   Dimension: $1$ to $9$
-   Tolerance: $0$