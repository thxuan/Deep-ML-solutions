import numpy as np
def compressed_row_sparse_matrix(dense_matrix):
	dense_matrix = np.array(dense_matrix)
	rows,cols = dense_matrix.shape
	Values = []
	indice = []
	row_pointer = []
	row_counter = 0
	for row in range(rows):
		if row_counter != 0:
			row_pointer.append(0)
		elif row_counter != 0:
			row_pointer.append(row_counter) 
			
		for col in range(cols):
			if dense_matrix[row][col] != 0:
				Values.append(dense_matrix[row][col].tolist()) 
				indice.append(col) 
				row_counter += 1
		if row == rows-1:
			row_pointer.append(row_counter)
	return Values,indice,row_pointer

