def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    if not a or not b:
        return -1
    rows_a = len(a)
    rows_b = len(b)
    cols_a = len(a[0])
    cols_b = len(b[0])
    for row in a:
        if len(row) != cols_a:
            return -1
    for row in b:
        if len(row) != cols_b:
            return -1
    if cols_a != rows_b or rows_a != cols_b:
        return -1

    c = [[] for _ in range(rows_a)]
    sum = 0
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(rows_b):
                sum += a[i][k] * b[k][j]
                if k == rows_b - 1:
                    c[i].append(sum)
                    sum = 0
    
    return c