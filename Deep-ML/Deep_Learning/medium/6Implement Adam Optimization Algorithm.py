import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Your code here
    mt = 0
    mt_hat = 0

    vt = 0
    vt_hat = 0
    for iteration in range(num_iterations):
        mt = beta1*mt + (1-beta1)*grad(x0)
        vt = beta2*vt + (1-beta2)*(grad(x0)**2)
        mt_hat = mt/(1-(beta1)**(iteration+1))
        vt_hat = vt/(1-(beta2)**(iteration+1))
        x0 -= learning_rate*mt_hat/(np.sqrt(vt_hat)+epsilon)
    return x0 




def objective_function(x):
    return x[0]**2 + x[1]**2

def gradient(x):
    return np.array([2*x[0], 2*x[1]])

x0 = np.array([1.0, 1.0])
x_opt = adam_optimizer(objective_function, gradient, x0)

print("Optimized parameters:", x_opt)