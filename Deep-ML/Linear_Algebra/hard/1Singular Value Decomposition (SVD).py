import numpy as np 
def jacobi_rotation(B: np.ndarray) -> tuple:
    if B[0][0] == B[1][1]:
        theta = (np.pi)/4
        r = np.array([ 
            [np.cos(theta),-np.sin(theta)],
            [np.sin(theta), np.cos(theta)] 
            ])
        r_T = r.T
        d = r_T.dot(B).dot(r)
    else:
        # theta = 0.5 * np.arctan2(2 * B[0,1], B[0,0] - B[1,1])
        theta = (np.arctan2( (2*B[0][1]),(B[0][0]-B[1][1]) ))/2
        r = np.array([ 
            [np.cos(theta),-np.sin(theta)],
            [np.sin(theta), np.cos(theta)] 
            ])
        r_T = r.T
        d = r_T.dot(B).dot(r)

    return np.round(d,4),r

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    A = np.array(A)
    d,v= jacobi_rotation( (A.T).dot(A) ) 
    # s = np.sort( np.sqrt([d[0][0],d[1][1]]) )[::-1]
    s = np.sqrt([d[0][0],d[1][1],]) 
    s_inv = np.array([ 
        [1/s[0],0],
        [0,1/s[1]]
                       ])
    u = A.dot(v).dot(s_inv)
    return u,s,v.T



print(svd_2x2_singular_values([
    [-10, 8],
    [10, -1]
      ]))