import numpy as np 


def max_pool_forward(x, conv_param):

    kernel = conv_param["kernel"]
    stride = conv_param["stirde"]
    batch_size, in_channel, h, w = x.shape 
    kh, kw= kernel

    out_h = 1 + (h - kh) // stride 
    out_w = 1 + (w - kw) // stride 

    output = np.zeros((batch_size, in_channel,out_h, out_w))

    for bs in range(batch_size):
        for ic in range(in_channel):
            for i in range(out_h):
                for j in range(out_w):
                    region = x[bs, ic, i*stride:i*stride + kh, j*stride:j*stride + kw]
                    output[bs, ic, i, j] = np.max(region)
    cache = (x, kernel, stride) 

    return output, cache 

def max_pool_backward(dout, cache):

    x , kernel ,stride = cache
    kh, kw = kernel 
    batch_size , in_channel, h, w = x.shape
    _, _, out_h, out_w = dout.shape

    dx = np.zeros_like(x)

    for bs in range(batch_size):
        for ic in range(in_channel):
            for i in range(out_h):
                for j in range(out_w):
                    region = x[bs, ic, i*stride:i*stride +kh, j*stride:j*stride +kw]
                    max_val = np.max(region)   
                    mask = (region ==  max_val)

                    dx[bs, ic, i*stride:i*stride + kh, j*stride:j*stride + kw] += mask * dout[bs, ic, i, j]
    return dx 