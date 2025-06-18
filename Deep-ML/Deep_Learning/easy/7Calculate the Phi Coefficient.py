import numpy as np
def phi_corr(x: list[int], y: list[int]) -> float:
    x = np.array(x)
    y = np.array(y)
    x00 = sum( (x == 0) & (y == 0) )
    x01 = sum( (x == 0) & (y == 1) )
    x10 = sum( (x == 1) & (y == 0) )
    x11 = sum( (x == 1) & (y == 1) )
    val = ( (x00 * x11) - (x01 * x10) ) / np.sqrt( (x00 + x01) * (x00 + x10) * (x11 + x01) * (x11 + x10) )
    return round(val,4)

print(phi_corr([1, 1, 0, 0], [0, 0, 1, 1]))