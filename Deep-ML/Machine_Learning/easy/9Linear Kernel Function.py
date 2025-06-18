import numpy as np
def kernel_function(x1, x2):
	# Your code here
	x1 = np.array(x1)
	x2 = np.array(x2)
	return np.dot(x1,x2)

print(kernel_function(
	[1,2,3],
	[4,5,6]
))
