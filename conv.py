import numpy as np 

def conv_forward(x, W, b, conv_param):
        
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    batch_size , in_channel, h, w = x.shape
    out_channel, _, kh, kw = W.shape
    out_h = 1 + (h + 2 * pad - kh) // stride     
    out_w = 1 + (w + 2 * pad - kw) // stride
    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant')
    output = np.zeros((batch_size, out_channel, out_h, out_w))
    for bs in range(batch_size):
        for oc in range(out_channel):
            for i in range(out_h):
                for j in range(out_w):
                    region_sum = 0
                    for ic in range(in_channel):
                        region = x_pad[bs, ic, i*stride: i *stride + kh, j*stride: j*stride + kw]
                        region_sum += np.sum(region * W[oc, ic]) 
        
                    output[bs, oc, i, j] = region_sum + b[oc]
    
    cache = (x, W, b, stride, pad)
    return output, cache
    
def conv_backward(dout, cache):

    x, W, b, stride, pad =  cache
    batch_size, out_channel, out_h, out_w = dout.shape 
    _, in_channel, h, w = x.shape
    _, _, kh, kw = W.shape
    dx = np.zeros_like(x)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    dx_pad = np.pad(dx, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant')
    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant')
    db = np.sum(dout, axis = (0, 2,3))
    for bs in range(batch_size):
        for oc in range(out_channel):
            for i in range(out_h):
                for j in range(out_w):
                    for ic in range(in_channel):
                        region =  x_pad[bs,ic, i*stride:i*stride+kh, j*stride: j*stride +kw]
                        dW[oc, ic] += region * dout[bs, oc, i, j]
                        dx_pad[bs, ic, i*stride: i*stride + kh, j*stride:j*stride + kw] += W[oc,ic] * dout[bs, oc, i, j]
    dx = dx_pad[:,:,pad:pad + h, pad:pad+w]
    return dx, dW, db
    