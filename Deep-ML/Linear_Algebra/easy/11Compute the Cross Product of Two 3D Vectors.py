import numpy as np
def cross_product(a, b):
    a = np.array(a)
    b = np.array(b)
    c = np.cross(a,b)
    return c

print(cross_product([1, 0, 0], [0, 1, 0]))