import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    X = np.array(X)
    b=[]
    m1=np.hstack( ( np.ones((X.shape[0],1)),X ) )
    m2=[[] for _ in range(X.shape[0])]
    degree_base = 2
    diff = degree -degree_base
    for i in range(X.shape[0]):
        degree_base=2
        if diff==0:
            for a in combinations_with_replacement(X[i],degree):
                v = 1
                for n in range(degree):
                    v=v*a[n]
                m2[i].append(v)
        else:
            for j in range(diff+1):
                degree_base += j
                for a in combinations_with_replacement(X[i],degree_base):
                    v = 1
                    for n in range(degree_base):
                        v=v*a[n]
                    m2[i].append(v)

    m = np.hstack( (m1,m2) )
    return m

print(polynomial_features(np.array([[1, 2, 3], [3, 4, 5], [5, 6, 9]]), 3))           
# print(polynomial_features(np.array([[1, 2], [3, 4], [5, 6]]), 3))

# print(polynomial_features(
#     X = np.array([[2, 3],
#                   [3, 4],
#                   [5, 6]]),
#     degree = 2
# ))

import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    n_samples, n_features = np.shape(X)

    # Generate all combinations of feature indices for polynomial terms
    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    
    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))
    
    # Compute polynomial features
    for i, index_combs in enumerate(combinations):  
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new
    