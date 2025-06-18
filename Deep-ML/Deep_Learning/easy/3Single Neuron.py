import numpy as np
def single_neuron_model(features: list[list[float]], \
						labels: list[int], \
						weights: list[float], \
						bias: float) -> tuple[(list[float], float)]:
	# Your code here
	np_features = np.array(features)
	np_weights = np.array(weights)
	z = (np_features * np_weights).sum(axis= 1) + bias
	predict_prob = np.round(1/(1 + np.exp(-z)),4)
	mse = 1/len(predict_prob) * (sum( (predict_prob - labels) ** 2 ))
	mse = float(np.round(mse,4))
	return predict_prob.tolist(),mse

print(single_neuron_model(features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], \
						  labels = [0, 1, 0], weights = [0.7, -0.4], bias = -0.1))