import math
def elu(x: float, alpha: float = 1.0) -> float:
    if x > 0:
        return float(x)
    else:
        return round( alpha*(math.exp(x) - 1) , 4)