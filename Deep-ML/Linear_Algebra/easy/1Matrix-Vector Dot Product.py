def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
    if not a or not b:
        return -1
    
    rows = len(a)
    cols = len(a[0]) if a else 0

    for row in a:
        if len(row) != cols:
            return -1
    
    if cols != len(b):
        return -1
    
    result = []
    for i in range(rows):
        row_sum = 0
        for j in range(cols):
            row_sum += a[i][j] * b[j]
        result.append(row_sum)
    return result
pass