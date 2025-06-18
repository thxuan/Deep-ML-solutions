import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, 
				 initial_weights: np.ndarray,initial_bias: float, 
				 learning_rate: float, epochs: int)\
				 -> tuple[ np.ndarray, float, list[float] ]:
	# Your code here
	weights = initial_weights
	bias = initial_bias
	mse = []
	for epoch in range(epochs):
		z = np.dot(features,weights) + bias
		sigmoid_z = 1/(1+np.exp(-z))
		mse.append(np.round(np.mean((sigmoid_z  - labels)**2),4).tolist()) 
		der_sigmoid_z = np.exp(-z)/( (1+np.exp(-z)) **2 )
		weight_error =  (sigmoid_z - labels)*der_sigmoid_z*np.transpose(features) 
		bias_error =  (sigmoid_z - labels)*der_sigmoid_z
		weights -= learning_rate*2*np.mean(weight_error,axis=1)
		bias -= learning_rate*2*np.mean(bias_error)
	return np.round(weights,4).tolist(), round(bias.tolist(),4), mse



print(train_neuron(
	features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]], 
	labels = [1, 0, 0], 
	initial_weights = [0.1, -0.2], 
	initial_bias = 0.0, 
	learning_rate = 0.1, 
	epochs = 2
))