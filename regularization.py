import numpy as np


def dropout_forward(x, p):
    
    mask = (np.random.rand(*x.shape) < p) / p
    out = x * mask

    cache =  (p, mask)
    return out, cache 

def dropout_backward(dout, cache):

    p, mask = cache 
    dx = mask * dout
    return dx



