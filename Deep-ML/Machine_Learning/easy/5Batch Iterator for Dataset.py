import numpy as np
def batch_iterator(X, y=None, batch_size= int):
	# Your code here
	np_X = np.array(X)
	samples = len(np_X)
	if y is not None:
		np_y = np.array(y)

	folds = samples//batch_size + 1  if samples % batch_size != 0 else samples//batch_size
	foldsizes = np.full(folds, batch_size)
	foldsizes[ folds-1 ] = samples % batch_size if samples % batch_size != 0 else foldsizes[ folds-1 ]
	
	split = []
	current = 0
	for foldsize in foldsizes:
		array_X = np_X[current : current + foldsize].tolist()
		if np_y is not None:
			array_y = np_y[current: current + foldsize].tolist()
			split.append( (array_X,array_y) )
		else:
			split.append(array_X)
		current += foldsize
	return split

print(batch_iterator(X = np.array([[1, 2], 
                  				   [3, 4], 
                  				   [5, 6], 
                  				   [7, 8], 
                  				   [9, 10]]),
   					y = np.array([1, 2, 3, 4, 5]),
   					batch_size = 2))