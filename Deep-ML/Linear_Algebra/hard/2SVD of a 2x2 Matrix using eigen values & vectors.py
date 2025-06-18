import numpy as np

def poly(B:np.ndarray) -> tuple:
    delta =  (B[0,0]+B[1,1])**2 - 4*( B[0,0]*B[1,1] - B[0,1]*B[1,0] ) 
    if delta <0:
        return 0
    else:
        delta = np.sqrt(delta)
        r1 = ( (B[0,0]+B[1,1])+delta )*0.5
        r2 = ( (B[0,0]+B[1,1])-delta )*0.5
        if r1 > r2:
            return r1,r2
        else:
            return r2,r1
    

def svd_2x2(A: np.ndarray) -> tuple:
    # Your code here
    A = np.array(A)
    b = A.dot(A.T)
    s = np.array(poly(b))
    u = [[],[]]

    x1 = -(b[0,1]/(b[0,0]-s[0])) 
    n = 1/np.sqrt( (x1**2) +1)
    x1 = n *x1
    x2 = n 
    u[0].append(x1)
    u[1].append(x2)

    x1 = -(b[0,1]/(b[0,0]-s[1])) 
    n = 1/np.sqrt( (x1**2) +1)
    x1 = n *x1
    x2 = n 
    u[0].append(x1)
    u[1].append(x2)

    sforinv = np.array([
        [np.sqrt(s[0]),0],
        [0,np.sqrt(s[1])]
    ])
    v = np.linalg.inv(sforinv).dot(np.linalg.inv(u)).dot(A)
    return np.round(u,4),np.sqrt(s),v
    # return u @ np.diag(np.sqrt(s)) @ v


print(svd_2x2(
    A = [[-10, 8], 
         [10, -1]]
))