import numpy as np 


def relu_forward(x):
    out = None
    out  = np.maximum(0,x)
    relu_cache = x
    return out, relu_cache 

def relu_backward(dout,cache):
    x = cache
    dx = (x > 0) * dout
    return dx 

def sigmoid(x):
    out = 1 / (1 + np.exp(-x))
    softmax_cache = out
    return out, softmax_cache

def sigmoid_backward(dout, cache):
    out = cache 
    dx = (out*(1 - out))* dout
    return dx  


