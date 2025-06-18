import numpy as np

def cramers_rule(A, b):
    A = np.array(A)
    b = np.array([b])
    det_A = np.linalg.det(A)
    if det_A == 0:
        return -1
    subsets_a = np.split(A,A.shape[1],axis=1)
    det_Ax = []
    for i in range(A.shape[1]):
        new_A = subsets_a[0] if i != 0 else subsets_a[1]
        for j in range(A.shape[1]-1):
            if j == i and i == 0:
                new_A = np.hstack( (b.T,new_A) )
            elif j+1 == i:
                new_A = np.hstack( (new_A,b.T) )
            else:
                new_A = np.hstack( (new_A,subsets_a[j+1]))
                
        det_Ax.append(np.linalg.det(new_A))
    x = det_Ax/det_A
    return(np.round(x,4))




print(cramers_rule(A = [[2, -1, 3], 
                        [4, 2, 1], 
                        [-6, 1, -2]], 
                        b = [5, 10, -3]))