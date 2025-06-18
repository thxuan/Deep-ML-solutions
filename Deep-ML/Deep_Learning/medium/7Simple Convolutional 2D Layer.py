import numpy as np

def simple_conv2d(input_matrix: np.ndarray, 
                  kernel: np.ndarray, 
                  padding: int, 
                  stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape
    for _ in range(padding):
        input_matrix = np.vstack( (np.zeros( input_width ),input_matrix,np.zeros( input_width )) )
        input_height += 2
        input_matrix = np.hstack( (np.zeros( (input_height,1) ),input_matrix,np.zeros( (input_height,1) )) )
        input_width += 2

    output_height = ( (input_height - kernel_height)//stride ) + 1
    output_width =  ( (input_width - kernel_width)//stride ) + 1
    output_matrix = [[] for _ in range(output_height)]

    for i in range(output_height):
        row_indice = i*stride
        for j in range(output_width):
            col_indice = j*stride
            output_matrix[i].append(np.sum(input_matrix[row_indice : kernel_height+row_indice ,col_indice : kernel_width+col_indice] * kernel).tolist()) 
            #print(f'output_matrix{i,j}:{input_matrix[row_indice : kernel_height+row_indice ,col_indice : kernel_width+col_indice]}')

    return output_matrix


input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])
kernel = np.array([
    [1, 0],
    [-1, 1]
])
padding = 1
stride = 2

output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)
