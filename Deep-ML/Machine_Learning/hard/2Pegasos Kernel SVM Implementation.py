import numpy as np
def rbf(dataset:np.ndarray,
        data:np.ndarray,
        sigma) ->np.ndarray: 
    return np.exp(- ((dataset-data)**2).sum(axis = 1) / (2* (sigma**2)))

def pegasos_kernel_svm(data: np.ndarray, 
                       labels: np.ndarray, 
                       kernel='linear', 
                       lambda_val=0.01, 
                       iterations=100,
                       sigma=1.0) -> tuple:
    # Your code here
    n = len(data)
    alphas = np.zeros(n)
    b = 0 
    f_x = []

    for iteration in range(iterations):

        if kernel == 'linear':
            learning_rate = 1/(lambda_val*(iteration+1))
            for i in range(n):
                f_x = np.sum(alphas * labels* (data @ data[i]))+b          
                if f_x*labels[i] < 1:
                    alphas[i] += learning_rate*(labels[i] - lambda_val*alphas[i])
                    b += learning_rate*labels[i]

        if kernel == 'rbf':
            learning_rate = 1/(lambda_val*(iteration+1))
            for i in range(n):
                f_x = np.sum(alphas * labels* rbf(data,data[i],sigma))+b          
                if f_x*labels[i] < 1:
                    alphas[i] += learning_rate*(labels[i] - lambda_val*alphas[i])
                    b += learning_rate*labels[i]

    return alphas, b



# print(pegasos_kernel_svm(
#     data = np.array([[1, 2], [2, 3], [3, 1], [4, 1]]),
#     labels = np.array([1, 1, -1, -1]), 
#     kernel = 'linear',
#     lambda_val = 0.01, 
#     iterations = 3,
#     sigma = 1.0
# ))

print(pegasos_kernel_svm(np.array([[1, 2], [2, 3], [3, 1], [4, 1]]), 
                         np.array([1, 1, -1, -1]), 
                         kernel='rbf', 
                         lambda_val=0.01, 
                         iterations=100, 
                         sigma=0.5))

# print(pegasos_kernel_svm(
#     data = np.array([[1, 2], [2, 3], [3, 1], [4, 1]]), 
#     labels = np.array([1, 1, -1, -1]), 
#     kernel = 'rbf', 
#     lambda_val = 0.01, 
#     iterations = 100, 
#     sigma = 1.0
# ))


