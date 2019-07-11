import numpy as np
import matplotlib.pyplot as plt
import math

def Local_weight_generator_RF(input_size, output_size, RF):
    input_range = 1.0 / input_size ** (1/2)
    padding = ((RF - 1) // 2)
    w = np.zeros(shape=(input_size + 2*padding, output_size))
    M = np.zeros(shape=(input_size + 2*padding, output_size))
    step = float(w.shape[0] - RF) / (output_size - 1)
    for i in range(output_size):
        j = int(math.ceil(i * step))
        j_next = j + RF
        w[j:j_next, i] = np.random.normal(loc=0, scale=input_range, size=(j_next-j))
        M[j:j_next, i] = 1
        weight_mat = w[padding:-padding, :]
        Mask_mat = M[padding:-padding, :]
    return weight_mat, Mask_mat

mat, mas = Local_weight_generator_RF(65*65,33*33,5)
# Display mask matrix
plt.matshow(mas)
plt.show()

# locally connected 2D layer with 3x3 kernel (input/hidden with odd size only)
# Mecha_cortex structure : 257-> 129-> 65-> 33-> 17-> 9-> 5
def L2D_weight(input_size, output_size):
    # init kernel center position in the input
    # kernel size = 3x3, so for each position we need to get the element :
    # [[kcl,kcc)],[kcl,kcc+1)],[kcl-1,kcc+1],[kcl-1,kcc],[kcl-1,kcc-1],[kcl,kcc-1],[kcl+1,kcc-1],[kcl+1,kcc],[kcl+1,kcc+1]]
    # in a case of a inhibitory matrix we have the same position except the center
    kcl = 0 # line
    kcc = 0 # column
    # list containing all possible movement
    kernel = [[kcl,kcc)],[kcl,kcc+1)],[kcl-1,kcc+1],[kcl-1,kcc],[kcl-1,kcc-1],[kcl,kcc-1],[kcl+1,kcc-1],[kcl+1,kcc],[kcl+1,kcc+1]]
    # calculate input range
    input_range = 1.0 / (input_size**2) ** (1/2)
    # create a binary mask and weight matrix
    mask = np.zeros((input_size**2, output_size**2))
    weight = np.zeros((input_size**2, output_size**2))
    # init hidden number value
    hidden_index = 0

    # sliding kernel over the dummy input + map the 2d pose to a 1D vector
    # stride = 2 (kernel center to kernel center)
    for u in range(0,input_size,2):
        for v in range(0,input_size,2):
            kcl = u
            kcc = v
            for elem in range(len(kernel)):
                if kernel[elem][0] > input_size or kernel[elem][0] < 0 or kernel[elem][1] > input_size or kernel[elem][1] < 0:
                    pass
                else:
                    # calculate the position in the mask matrix
                    line = kernel[elem][0]
                    column = kernel[elem][1]
                    input_index =

                    mask[input_index, hidden_index] = 1
        hidden_index += 1

    # fill the weight matrix with init weight
    for i in range(input_size):
        for j in range(output_size):
            if mask[i,j] == 1:
                weight[i,j] = np.random.normal(loc=0, scale=input_range)
            else:
                weight[i,j] = 0

    return weight, mask
