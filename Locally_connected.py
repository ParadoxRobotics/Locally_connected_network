import numpy as np
import matplotlib.pyplot as plt
import math

# locally connected 1D layer
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

mat, mas = Local_weight_generator_RF(81,25,5)

# locally connected 2D layer with 3x3 kernel (input/hidden with odd size only)
# Mecha_cortex structure : 257-> 129-> 65-> 33-> 17-> 9-> 5
def L2D_weight(input_size, output_size, inh):
    # init kernel center position in the input
    # kernel size = 3x3, so for each position we need to get the element :
    # [[kcl,kcc)],[kcl,kcc+1)],[kcl-1,kcc+1],[kcl-1,kcc],[kcl-1,kcc-1],[kcl,kcc-1],[kcl+1,kcc-1],[kcl+1,kcc],[kcl+1,kcc+1]]
    # in a case of a inhibitory matrix we have the same position except the center
    kcl = 0 # line
    kcc = 0 # column
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
            # verify type of connection
            if inh == True:
                # local inhibition
                kernel = [[kcl,kcc+1],[kcl-1,kcc+1],[kcl-1,kcc],[kcl-1,kcc-1],[kcl,kcc-1],[kcl+1,kcc-1],[kcl+1,kcc],[kcl+1,kcc+1]]
            else:
                # local excitation
                kernel = [[kcl,kcc],[kcl,kcc+1],[kcl-1,kcc+1],[kcl-1,kcc],[kcl-1,kcc-1],[kcl,kcc-1],[kcl+1,kcc-1],[kcl+1,kcc],[kcl+1,kcc+1]]
            # add each kernel to the weight and mask matrix
            for elem in range(len(kernel)):
                if kernel[elem][0] > input_size-1 or kernel[elem][0] < 0 or kernel[elem][1] > input_size-1 or kernel[elem][1] < 0:
                    pass
                else:
                    # calculate the position in the mask matrix
                    line = kernel[elem][0]
                    column = kernel[elem][1]
                    input_index = line*input_size + column
                    mask[input_index, hidden_index] = 1
            hidden_index += 1


    # fill the weight matrix with init weight
    for i in range(input_size**2):
        for j in range(output_size**2):
            if mask[i,j] == 1:
                weight[i,j] = np.random.normal(loc=0, scale=input_range)
            else:
                pass

    return weight, mask

w,m = L2D_weight(9,5,False)

# locally connected recurrent unit
def LR_weight(Hidden_size):
    W_rec = np.zeros((Hidden_size**2, Hidden_size**2))
    Mask_rec = np.zeros((Hidden_size**2, Hidden_size**2))
    for i in range(Hidden_size**2):
        W_rec[i,i] = np.random.uniform(0,1)
        Mask_rec[i,i] = 1
    return W_rec, Mask_rec

wr, mr = LR_weight(5)

# lateral inhibition weight matrix
def LI2D_weight(hidden_size):
    # init kernel center position in the input
    # kernel size = 3x3, so for each position we need to get the element :
    # [[kcl,kcc+1)],[kcl-1,kcc+1],[kcl-1,kcc],[kcl-1,kcc-1],[kcl,kcc-1],[kcl+1,kcc-1],[kcl+1,kcc],[kcl+1,kcc+1]]
    kcl = 0 # line
    kcc = 0 # column
    # calculate input range
    input_range = 1.0 / (hidden_size**2) ** (1/2)
    # create a binary mask and weight matrix
    mask = np.zeros((hidden_size**2, hidden_size**2))
    weight = np.zeros((hidden_size**2, hidden_size**2))
    # init hidden number value
    hidden_index = 0
    # sliding kernel over the dummy input + map the 2d pose to a 1D vector
    # stride = 1 (lateral connection in the hidden state)
    for u in range(0,hidden_size):
        for v in range(0,hidden_size):
            kcl = u
            kcc = v
            # local inhibition
            kernel = [[kcl,kcc+1],[kcl-1,kcc+1],[kcl-1,kcc],[kcl-1,kcc-1],[kcl,kcc-1],[kcl+1,kcc-1],[kcl+1,kcc],[kcl+1,kcc+1]]
            # add each kernel to the weight and mask matrix
            for elem in range(len(kernel)):
                if kernel[elem][0] > hidden_size-1 or kernel[elem][0] < 0 or kernel[elem][1] > hidden_size-1 or kernel[elem][1] < 0:
                    pass
                else:
                    # calculate the position in the mask matrix
                    line = kernel[elem][0]
                    column = kernel[elem][1]
                    input_index = line*hidden_size + column
                    mask[input_index, hidden_index] = 1
                    mask[hidden_index, input_index] = 1
            hidden_index += 1

    # fill the weight matrix with init weight
    for i in range(hidden_size**2):
        for j in range(hidden_size**2):
            if mask[i,j] == 1:
                weight[i,j] = np.random.normal(loc=0, scale=input_range)
            else:
                pass

    return weight, mask

wi,mi = LI2D_weight(17)
