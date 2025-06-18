import numpy as np
def softmax(values):
    return np.exp(values)/np.sum( np.exp(values),axis=1,keepdims=True)
    # Implement the softmax function

def pattern_weaver(n, crystal_values, dimension):
    # Your code here
    crystal_values = np.array([crystal_values])
    attention_score = (crystal_values * crystal_values.T)/np.sqrt(dimension)
    softmax_score = softmax(attention_score)
    final_value = softmax_score @ crystal_values.T
    return np.round(final_value.flatten(),3)

print(
    pattern_weaver(
    n=5,
    crystal_values=[4,2,7,1,9],
    dimension=1,
))