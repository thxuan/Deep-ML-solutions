def hard_sigmoid(x: float) -> float:
    if x >= 2.5:
        return 1.0
    elif x <= -2.5:
        return 0
    else:
        return (0.2*x+0.5)