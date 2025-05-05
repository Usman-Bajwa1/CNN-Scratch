import numpy as np 


def relu(x):
    out = None
    out  = np.maximum(0,x)
    relu_cache = x
    return out, relu_cache 

def relu_backward(dout,cache):
    dx, x = None 
    x = cache
    dx = (x > 0) * dout
    return dx 

def sigmoid(x):
    out = None
    out = 1 / (1 + np.exp(-x))
    softmax_cache = out
    return out, softmax_cache

def sigmoid_backward(dout, cache):
    dx, out = None
    out = cache 
    dx = (out*(1 - out))* dout
    return dx  


