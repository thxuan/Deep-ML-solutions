import numpy as np
def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	np_list = np.array(scores)
	max_score = np.argmax(np_list)
	a = np.log( sum( np.exp( np_list[j] - np_list[max_score]) for j in range(len(np_list)) ) )
	result = np_list - np_list[max_score] - a
	return np.round(result,4).tolist()

print(log_softmax(scores = np.array([1, 2, 3])))