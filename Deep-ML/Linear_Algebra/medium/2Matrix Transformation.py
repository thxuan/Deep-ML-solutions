import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:

    if not A or not T or not S:
        return -1
    
    a = np.array(A)
    t = np.array(T)
    if not np.linalg.det(t):
        return -1
    tInv = np.linalg.inv(t)
    s = np.array(S)
    if not np.linalg.det(s):
        return -1    

    tInva =  np.dot(a,tInv)
    transformed_matrix = np.dot(tInva,s)

    return transformed_matrix