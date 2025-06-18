def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix:
        return[]
    
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]
    inverse = [[],[]]
    if (a*d - b*c) == 0:
        return[]
    
    det = a*d - b*c
    in_a = 1/det*d
    in_b = -1/det*b
    in_c = -1/det*c
    in_d = 1/det*a
    inverse[0].append(in_a)
    inverse[0].append(in_b)
    inverse[1].append(in_c)
    inverse[1].append(in_d)
    return inverse
    
