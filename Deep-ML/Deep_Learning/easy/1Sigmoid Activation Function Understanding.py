import math
def sigmoid(z: float) -> float:
	#Your code here
	result = 1 / (1 + math.exp(-z))
	return round(result,4)

print(sigmoid(0))
