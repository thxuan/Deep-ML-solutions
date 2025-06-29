import numpy as np
def compressed_col_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
	dense_matrix = np.array(dense_matrix)
	values = []
	row_indeces = []
	col_pointers = [0]
	for col in dense_matrix.T:
		for index,val in enumerate(col):
			if val != 0:
				values.append(val.tolist())
				row_indeces.append(index)
		col_pointers.append(len(values))
	return values,row_indeces,col_pointers


dense_matrix = [ [0, 0, 0], [1, 2, 0], [0, 3, 4] ] 
vals, row_idx, col_ptr = compressed_col_sparse_matrix(dense_matrix) 
print(vals)