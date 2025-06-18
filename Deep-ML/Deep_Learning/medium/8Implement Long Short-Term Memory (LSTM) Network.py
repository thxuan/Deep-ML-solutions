import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        time = x.shape[0]
        hidden_state = initial_hidden_state
        c = initial_cell_state

        for t in range(time):
            a = np.vstack( (hidden_state,x[t].reshape(-1, 1)))
            f_t = sigmoid( (self.Wf @ a)+self.bf )

            i_t = sigmoid( (self.Wi @ a)+self.bi )
            c_hat_t = tanh( (self.Wc @ a)+self.bc )

            c_t = (f_t * c) + (i_t * c_hat_t)

            o_t = sigmoid( (self.Wo @ a)+self.bo )
            hidden_state = o_t * tanh(c_t)
            
            c = c_t


        return o_t, hidden_state,c_t



input_sequence = np.array([[0.1, 0.2], [0.3, 0.4]]) 
initial_hidden_state = np.zeros((2, 1)) 
initial_cell_state = np.zeros((2, 1)) 
lstm = LSTM(input_size=2, hidden_size=2) 
# Set weights and biases for reproducibility 
lstm.Wf = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]) 
lstm.Wi = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]) 
lstm.Wc = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]) 
lstm.Wo = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]) 
lstm.bf = np.array([[0.1], [0.2]]) 
lstm.bi = np.array([[0.1], [0.2]]) 
lstm.bc = np.array([[0.1], [0.2]]) 
lstm.bo = np.array([[0.1], [0.2]]) 
#outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state) 
print(lstm.forward(input_sequence, initial_hidden_state, initial_cell_state) )




# input_sequence = np.array([[1.0], [2.0], [3.0]])
# initial_hidden_state = np.zeros((1, 1))
# initial_cell_state = np.zeros((1, 1))

# lstm = LSTM(input_size=1, hidden_size=1)
# #outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)
# print( lstm.forward(input_sequence, initial_hidden_state, initial_cell_state))

