def min_max(x: list[int]) -> list[float]:
    result = []
    for i in range(len(x)):
        scale = max(x) - min(x)
        if scale !=0:
            result.append( (x[i] - min(x)) / scale ) 
        else:
            result.append(0.0)
    return result

print(min_max([1, 2, 3, 4, 5]))