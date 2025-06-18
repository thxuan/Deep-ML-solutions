def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    # Your code here
    if not vectors:
        return []
    cols =len(vectors[0])
    for row in vectors:
        if len(row) != cols:
            return []
    cov_sum = 0
    rows = len(vectors)
    means = [sum(vector) / cols for vector in vectors]

    Covariance_Matrix = [[] for _ in range(rows)]
    for i in range(rows):
        for j in range(rows):
            for k in range(cols):
                cov_sum +=(vectors[i][k] - means[i]) * (vectors[j][k] - means[j])
                if k == cols-1:
                    Covariance_Matrix[i].append(cov_sum/(cols-1))
                    cov_sum = 0
    
    return Covariance_Matrix
