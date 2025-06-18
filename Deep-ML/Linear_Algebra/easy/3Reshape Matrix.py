import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
    if not a:
        return []
	
    cols = len(a[0])
    for row in a:
        if len(row) != cols:
            return []

    rows = len(a)
    total = rows * cols

    new_rows,new_cols = new_shape
    new_total = new_rows * new_cols
    if total != new_total:
        return []
    
    array = np.array(a)
    reshaped_array = array.reshape(new_rows, new_cols)
    reshaped_matrix = reshaped_array.tolist()
    
    return reshaped_matrix
