import numpy as np

def adam_optimizer(parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using the Adam optimizer.
    Adjusts the learning rate based on the moving averages of the gradient and squared gradient.
    :param parameter: Current parameter value
    :param grad: Current gradient
    :param m: First moment estimate
    :param v: Second moment estimate
    :param t: Current timestep
    :param learning_rate: Learning rate (default=0.001)
    :param beta1: First moment decay rate (default=0.9)
    :param beta2: Second moment decay rate (default=0.999)
    :param epsilon: Small constant for numerical stability (default=1e-8)
    :return: tuple: (updated_parameter, updated_m, updated_v)
    """
    # Your code here
    m_prev = m
    v_prev = v
    parameter_prev = parameter
    for i in range(1,t+1):
        m = beta1 * m_prev + (1 - beta1) * grad
        v = beta2 * v_prev + (1 - beta2) * (grad**2)
        mt_hat = m/(1 - beta1**i)
        vt_hat = v/(1 - beta2**i)
        parameter = parameter_prev - (learning_rate*mt_hat)/(np.sqrt(vt_hat)+epsilon)

        m_prev = m
        v_prev = v
        parameter_prev = parameter

    return np.round(parameter,5), np.round(m,5), np.round(v,5)

print(adam_optimizer(
    parameter = 1.0, 
    grad = 0.1, 
    m = 0.0, 
    v = 0.0, 
    t = 1
))