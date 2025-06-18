import numpy as np
def translate_object(points, tx, ty):
    points = np.array(points)
    a = np.array(points[:,0]+tx)
    b = np.array(points[:,1]+ty)
    c = np.vstack( (a,b))
    return c.T


points = [[0, 0], [1, 0], [0.5, 1]]
tx, ty = 2, 3

print(translate_object(points, tx, ty))