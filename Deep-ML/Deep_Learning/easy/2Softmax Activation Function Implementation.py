import math
def softmax(scores: list[float]) -> list[float]:
	# Your code here
	probabilities = []
	total = sum(math.exp(i) for i in scores)
	for score in scores:
		probabilities.append( round(math.exp(score) / total,4) )
	return probabilities

print(softmax(scores = [1, 2, 3]))