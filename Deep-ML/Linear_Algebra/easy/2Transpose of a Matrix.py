#Write a Python function that computes the transpose of a given matrix.
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    if not a:
        return -1
    else:
        rows = len(a)
        cols = len(a[0])
        for row in a:
            if len(row) != cols:
                return -1
    
    b = [[] for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            b[j].append(a[i][j])
    
    return b
