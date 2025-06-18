def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    if not matrix:
        return []
    
    rows = len(matrix)
    cols = len(matrix[0])
    for row in matrix:
        if len(row) != len(matrix[0]):
            return []
    
    result = [[] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            result.append(matrix[i][j] * scalar) 
    return result