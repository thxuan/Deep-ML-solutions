import numpy as np

def rnn_forward(input_sequence: list[list[float]], 
                initial_hidden_state: list[float], 
                Wx: list[list[float]], 
                Wh: list[list[float]], 
                b: list[float]) -> list[float]:
    x = np.array(input_sequence)
    initial_hidden_state = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)
    h = initial_hidden_state

    for i in range(x.shape[0]):
        z = x[i]@Wx.T + h@Wh.T + b
        h = ( np.exp(z)-np.exp(-z) ) /  ( np.exp(z)+np.exp(-z) )
    return np.round(h,4) 

print(rnn_forward( 
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], 
    [0.0, 0.0], 
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], 
    [[0.7, 0.8], [0.9, 1.0]], 
    [0.1, 0.2] ))


# print(rnn_forward(
#     input_sequence = [[1.0], [2.0], [3.0]],
#     initial_hidden_state = [0.0],
#     Wx = [[0.5]],  # Input to hidden weights
#     Wh = [[0.8]],  # Hidden to hidden weights
#     b = [0.0]     # Bias
#     ))