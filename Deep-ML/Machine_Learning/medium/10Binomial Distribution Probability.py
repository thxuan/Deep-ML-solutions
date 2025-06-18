import math

def binomial_probability(n, k, p):
	"""
    Calculate the probability of achieving exactly k successes in n independent Bernoulli trials,
    each with probability p of success, using the Binomial distribution formula.
    """
	value = 1
	for i in range(k):
		value *= (n-i)
	C = value/math.factorial(k)
	probability = C * (p**k) * ((1-p)**(n-k))
	# Your code here
	return round(probability, 5)

print(binomial_probability(n = 6, 
						   k = 2, 
						   p = 0.5))