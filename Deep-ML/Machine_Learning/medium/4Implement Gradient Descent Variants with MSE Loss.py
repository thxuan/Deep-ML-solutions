import numpy as np
def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
	# Your code here
	m = len(X)
	if method == 'batch':
		for iteration in range(n_iterations):
			weights -= learning_rate*2/m*( X.T.dot(X.dot(weights)-y) )
			# weights -= learning_rate*2*np.mean( ((weights*X).sum(axis=1) -y)*X.T, axis=1 )		
			
	elif method == 'stochastic':
		for iteration in range(n_iterations):
			for i in range(m):
				x_i = X[i]
				y_i = y[i]
				weights -= learning_rate*2*( x_i.dot( (weights.dot(x_i) -y_i )) )
			
	elif method == 'mini_batch':
		for iteration in range(n_iterations):
			for i in range(0,m,batch_size):
				# indices = np.random.choice(m,size=batch_size,replace=False)
				x_batch = X[i:i+batch_size]
				y_batch = y[i:i+batch_size]
				weights -= learning_rate*2/batch_size*( x_batch.T.dot(x_batch.dot(weights)-y_batch) )
						
	return weights

print(gradient_descent(
	X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]]),
    y = np.array([2, 3, 4, 5]),
	weights = np.zeros(2),
    learning_rate = 0.01,
    n_iterations = 100,
    batch_size = 2,
	method='mini_batch'
))

# print(gradient_descent(
# 	X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]]),
#     y = np.array([2, 3, 4, 5]),
# 	weights = np.zeros(2),
#     learning_rate = 0.01,
#     n_iterations = 1000,
#     batch_size = 2
# ))

