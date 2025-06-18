import numpy as np

def train_logreg(X: np.ndarray, 
                 y: np.ndarray, 
                 learning_rate: float, 
                 iterations: int) -> tuple[list[float], ...]:
    X = np.array(X)
    y = np.array(y)
    X = np.hstack( (np.ones( (X.shape[0],1) ),X) )
    omega = np.zeros((X.shape[1]))
    loss_value = []
    for iteration in range(iterations):
        z = X@omega
        p = 1/(1+np.exp(-z))
        loss = -np.sum( (y*np.log(p) + (1-y)*np.log(1-p)) )
        loss_value.append(np.round(loss,4))
        omega -= learning_rate*( X.T @ (p-y)  )
    return omega.T.round(4).tolist(),loss_value


print(train_logreg(np.array([[1.0, 0.5], [-0.5, -1.5], [2.0, 1.5], [-2.0, -1.0]]),
                   np.array([1, 0, 1, 0]),
                   0.01, 
                   20))