import numpy as np


def affine_forward(x, w, b):
    num_train = x.shape[0]
    x_reshaped = x.reshape(num_train,-1)

    out = np.dot(x_reshaped, w) + b
    fn_cache = x,w,b
    return out, fn_cache

def affine_backward(dout, cache):

    x,w,b = cache 
    num_train = x.shape[0]
    x_reshaped = x.reshape(num_train, -1)

    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x_reshaped.T, dout)
    db = np.sum(dout, axis = 0)
    return dx, dw, db

def softmax_loss(x, y):

    loss, dx = None, None

    num_train = x.shape[0]
    
    scores = x - np.max(x, axis = 1, keepdims= True)
    scores_exp = np.exp(scores)
    softmax_prob = scores_exp / np.sum(scores_exp, axis = 1, keepdims=True)

    loss = -np.log(softmax_prob[(np.arange(num_train)), y])
    loss = np.sum(loss) / num_train

    softmax_prob[(np.arange(num_train)), y] -= 1
    dx = softmax_prob / num_train 

    return loss, dx 

 

    







