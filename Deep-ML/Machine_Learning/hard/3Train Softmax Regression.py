import numpy as np
def train_softmaxreg(X: np.ndarray, 
                     y: np.ndarray, 
                     learning_rate: float, 
                     iterations: int) -> tuple[list[float], ...]:
    X = np.array(X)
    y = np.array(y)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    n_features = X.shape[1]
    n_labels = y.max()+1
    y = np.eye(n_labels)[y]
    omega = np.zeros( (n_features,n_labels) )
    loss_value = []


    for iteration in range(iterations):
        z = X @ omega
        p = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

        loss = -np.sum(np.log(p)[list(range(len(p))),np.argmax(y,axis=1)] )       
        loss_value.append(np.round(loss,4))
        omega -= learning_rate*( X.T @ (p-y) )

        
    return omega.T.round(4).tolist(),loss_value

print(train_softmaxreg(np.array([[0.5, -1.2], [-0.3, 1.1], [0.8, -0.6]]), 
                       np.array([0, 1, 2]), 
                       0.01, 
                       10
                       ))