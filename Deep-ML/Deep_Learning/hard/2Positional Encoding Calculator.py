import numpy as np
def encode_pos(j,value,position):
    if j%2 == 0:
        return np.sin(position/value)
    else:
        return np.cos(position/value)
    
def pos_encoding(position: int, d_model: int):
    # Your code here
    if position <= 0 or d_model <= 0 :
        return -1
    angle_rads = np.array( [ 10000**( 2*(j//2)/d_model ) for j in range(d_model) ] )
    en_pos = [[] for _ in range(position) ]
    for i in range(position):
        for j in range(d_model):
            en_pos[i].append( encode_pos( j,angle_rads[j],i ) ) 
    return np.float16(en_pos)


print(pos_encoding(
    position = 2, 
    d_model = 8
    ))