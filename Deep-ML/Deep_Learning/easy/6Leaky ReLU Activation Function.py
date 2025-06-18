def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
	# Your code here
	if z >= 0:
		return z
	else:
		return z*alpha
print(leaky_relu(-2, alpha=0.1))
	
