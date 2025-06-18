import math
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
  
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]

    eigenvalues = []
    coef_a = 1
    coef_b = a+d
    coef_c = a*d - b*c

    discriminant = coef_b*coef_b - 4*coef_a*coef_c
    if( discriminant < 0 ):
        return []
    else:
        eigenvalues.append((coef_b + math.sqrt(coef_b*coef_b - 4*coef_a*coef_c))/(2*coef_a))
        eigenvalues.append((coef_b - math.sqrt(coef_b*coef_b - 4*coef_a*coef_c))/(2*coef_a))

    return eigenvalues

        
    