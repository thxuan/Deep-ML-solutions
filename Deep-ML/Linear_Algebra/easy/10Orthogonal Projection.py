def orthogonal_projection(v, L):
    n = len(v)   
    t = sum( v[i]*L[i] for i in range(n)) / sum(L[j]**2 for j in range(n))
    v_project = []
    for k in range(n):
        v_project.append( round(t*L[k],3)  ) 
    return v_project

v = [3, 4]
L = [1, 0]
print(orthogonal_projection(v, L))