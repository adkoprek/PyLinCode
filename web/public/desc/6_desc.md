# mat_siz

## Task
Create a function named `mat_siz` that takes a matrix as its argument and returns its dimensions as a tuple of integers.

## Input
- `a: mat` - Input matrix

## Output
- `tuple[int, int]` - Tuple containing the number of rows and columns

## Example
$$
\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix} \rightarrow (2, 3)
$$

## Test
```python
a: mat = [[1, 2, 3], [4, 5, 6]]
result = mat_siz(a)
assert result == (2, 3)
```

## Cases
-   Test Cases: $50$
-   Dimension: $1$ to $10$
-   Tolerance: $0$