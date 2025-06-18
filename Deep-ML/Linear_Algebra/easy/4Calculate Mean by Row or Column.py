def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if not matrix:
        return []
    
    rows = len(matrix)
    cols = len(matrix[0])
    for row in matrix:
        if len(row) != cols:
            return []
    
    means = []
    mean = 0
    sum = 0
    if mode == "row":
        for i in range(rows):
            for j in range(cols):
                sum += matrix[i][j]
                if j == cols-1:
                    mean = sum/cols
                    means.append(mean)
                    sum = 0

    elif mode == "column":
        for i in range(cols):
            for j in range(rows):
                sum += matrix[j][i]
                if j == rows-1:
                    mean = sum/rows
                    means.append(mean)
                    sum = 0
    return means