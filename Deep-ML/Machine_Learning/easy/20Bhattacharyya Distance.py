import numpy as np
def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    if not p or not q:
        return 0.0
    if len(p) != len(q):
        return 0.0
    p = np.array(p)
    q = np.array(q)
    Bhattacharyya_Coefficient = np.sqrt(p*q).sum()
    Bhattacharyya_Distance = -np.log(Bhattacharyya_Coefficient)
    return np.round(Bhattacharyya_Distance,4)

print(bhattacharyya_distance(
    p = [0.1, 0.2, 0.3, 0.4], 
    q = [0.4, 0.3, 0.2, 0.1]
))