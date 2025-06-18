import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	np_B = np.array(B)
	np_C = np.array(C)
	Inv_C = np.linalg.inv(np_C)
	# P = np_B * Inv_C
	return np.round(Inv_C,4)

print(transform_basis(
	    B = [[1, 0, 0], 
             [0, 1, 0], 
             [0, 0, 1]],
        C = [[1, 2.3, 3], 
             [4.4, 25, 6], 
             [7.4, 8, 9]]
))