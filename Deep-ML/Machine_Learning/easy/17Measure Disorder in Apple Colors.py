import numpy as np
def disorder(apples: list) -> float:
    apples = np.array(apples)
    colors = np.unique(apples)
    base = 1
    for color in colors:
        n = (apples == color).sum()
        probabilities = n/apples.size
        base = base - probabilities ** 2
    return round(base,4)

print(disorder([1,1,0,0]))