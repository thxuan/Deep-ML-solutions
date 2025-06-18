import numpy as np
def softmax(S):
    return np.exp(S)/np.sum( np.exp(S),axis=1,keepdims=True )

def self_attention(Q, K, V):
    dk = K.shape[-1]
    S = Q@K.T/(np.sqrt(dk))
    A = softmax(S)@V
    return A

def compute_qkv(X,wq,wk,wv):
    X = np.array(X)
    wq = np.array(wq)
    wk = np.array(wk)
    wv = np.array(wv)
    Q = X @ wq
    K = X @ wk
    V = X @ wv
    return Q,K,V


X = np.array([[1, 0], [0, 1]]) 
W_q = np.array([[1, 0], [0, 1]]) 
W_k = np.array([[1, 0], [0, 1]]) 
W_v = np.array([[1, 2], [3, 4]]) 
Q, K, V = compute_qkv(X, W_q, W_k, W_v) 
output = self_attention(Q, K, V) 
print(output)


# X = np.array([[1, 0], [0, 1]])
# W_q = np.array([[1, 0], [0, 1]])
# W_k = np.array([[1, 0], [0, 1]])
# W_v = np.array([[1, 2], [3, 4]])

# print(compute_qkv(X, W_q, W_k, W_v))

