import numpy as np 
from conv import *
from activation import *
from maxppol import *
from fc import *
from regularization import *


def conv_relu_maxpool_forward(x, w, b, conv_param, pool_param):
    out1, conv_cache = conv_forward(x,w,b, conv_param)
    out2, relu_cache = relu_forward(out1)
    out, maxpool_cache = max_pool_forward(out2, pool_param)
    cache = (conv_cache, relu_cache, maxpool_cache)

    return out, cache


def affine_relu_forward(x,w,b):
    out1 , fn_cache = affine_forward(x,w,b)
    out , relu_cache = relu_forward(out1)
    cache = (fn_cache, relu_cache)

    return out, cache 

def affine_relu_backward(dout, cache):
    fn_cache, relu_cache = cache
    dr = relu_backward(dout, relu_cache)
    dx,dw,db = affine_backward(dr, fn_cache)

    return dx, dw, db

def conv_relu_maxpool_backward(dout, cache):
    conv_cache, relu_cache, maxpool_cache = cache

    ds = max_pool_backward(dout, maxpool_cache)
    dr = relu_backward(ds, relu_cache)
    dx,dw,db = conv_backward(dr, conv_cache)

    return dx, dw, db