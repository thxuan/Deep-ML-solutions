import numpy as np
def gini_impurity(y):
	y = np.array(y)
	classes = set(y)
	n = len(y)
	Gini_im=0
	for cls in classes:
		Gini_im += ( (np.sum( y == cls)/n)**2 )
	return np.round((1 - Gini_im),3)

print(gini_impurity(y = [0, 1, 2, 2, 2, 1, 2]))