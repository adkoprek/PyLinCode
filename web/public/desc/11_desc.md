# mat_ide

## Task
Create a function named `mat_ide` that takes a size as its argument and returns the identity matrix of size $n \times n$.

## Input
- `n: int` - Size of the identity matrix

## Output
- `mat` - Identity matrix of size $n \times n$

## Example
$$
I_3 =
\begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}
$$

## Test
```python
assert mat_ide(3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
```

## Cases
-   Test Cases: $50$
-   Dimension: $1$ to $10$ (rows and columns)
-   Tolerance: $0$