import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    rows, cols = A.shape
    if rows != cols:
        return []
    dg_a = A.diagonal()
    x =  [0,0,0]
    
    for _ in range(n):
        x_new = [0,0,0]
        for i in range(rows):
            s = sum(A[i][j] * x[j] for j in range(cols) if j != i)
            x_new[i] = 1/dg_a[i] * (b[i] - s)
        x = x_new
    
    return np.round(x, 4).tolist()

## Standard solution
import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    d_a = np.diag(A)
    nda = A - np.diag(d_a)
    x = np.zeros(len(b))
    x_hold = np.zeros(len(b))
    for _ in range(n):
        for i in range(len(A)):
            x_hold[i] = (1/d_a[i]) * (b[i] - sum(nda[i]*x))
        x = x_hold.copy()
    return np.round(x,4).tolist()