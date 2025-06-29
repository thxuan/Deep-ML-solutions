import numpy as np

def soft_max(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum( np.exp(x), axis=-1, keepdims=True )

def compute_qkv(X, W_q, W_k, W_v):
    return X @ W_q, X @ W_k, X @ W_v

def self_attention(Q, K, V, dk):
    return soft_max( (Q@K.T)/np.sqrt(dk) ) @ V

def multi_head_attention(Q, K, V, n_heads):
    dk = int(n/n_heads)
    Q = np.split(Q,n_heads,axis=-1)
    K = np.split(K,n_heads,axis=-1)
    V = np.split(V,n_heads,axis=-1)
    mth = []
    for i in range(n_heads):
        head = self_attention(Q[i],K[i],V[i],dk)
        mth.append(head)
    mho = np.concatenate(mth,axis=-1)
    return mho

m, n = 6, 8 
n_heads = 4
np.random.seed(42) 
X = np.arange(m*n).reshape(m,n) 
X = np.random.permutation(X.flatten()).reshape(m, n) 
W_q = np.random.randint(0,4,size=(n,n)) 
W_k = np.random.randint(0,5,size=(n,n)) 
W_v = np.random.randint(0,6,size=(n,n)) 
Q, K, V = compute_qkv(X, W_q, W_k, W_v) 
# test multi-head attention 
print( multi_head_attention(Q, K, V, n_heads) )

# m, n = 6, 8 
# n_heads = 4 
# np.random.seed(42) 
# X = np.arange(m*n).reshape(m,n) 
# X = np.random.permutation(X.flatten()).reshape(m, n) 
# W_q = np.random.randint(0,4,size=(n,n)) 
# W_k = np.random.randint(0,5,size=(n,n)) 
# W_v = np.random.randint(0,6,size=(n,n)) 
# Q, K, V = compute_qkv(X, W_q, W_k, W_v) 
# print(multi_head_attention(Q, K, V, n_heads))

# m, n = 4, 4 
# n_heads = 2 
# np.random.seed(42) 
# X = np.arange(m*n).reshape(m,n) 
# X = np.random.permutation(X.flatten()).reshape(m, n) 
# W_q = np.random.randint(0,4,size=(n,n)) 
# W_k = np.random.randint(0,5,size=(n,n)) 
# W_v = np.random.randint(0,6,size=(n,n)) 
# Q, K, V = compute_qkv(X, W_q, W_k, W_v) 
# print(multi_head_attention(Q, K, V, n_heads))