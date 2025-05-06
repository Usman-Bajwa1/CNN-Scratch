import numpy as np 
from layers import *

class ThreeLayerConvNet():

    def __init__(
        self,
        input_dim = (3, 32, 32), 
        num_filters = 32, 
        filter_size = 7, 
        hidden_dim = 100, 
        num_classes = 10, 
        weight_scale = 1e-3, 
        reg = 0.0,
        dtype = np.float32
        ):
 
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim
        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters)

        H_pool = H // 2
        H_width = W // 2

        self.params['W2'] = np.random.randn(num_filters*H_pool,H_width) * weight_scale
        self.params['b2'] = np.zeros(hidden_dim)

        self.params['W3'] = np.random.randn(hidden_dim,num_classes) * weight_scale
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y = None):
        
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        filter_size = W1.shape[2]

        conv_param = {"stride": 1, "pad": 2}
        pool_param = {"kernel": (2,2), "stride": 2}

        scores = None 
        
        # Forward pass
        conv_out, conv_cache = conv_relu_maxpool_forward(X, W1, b1, conv_param, pool_param)
        fc_out, fc_relu_cache = affine_relu_forward(conv_out, W2, b2)
        scores, fc_cache = affine_forward(fc_out, W3, b3)

        if y is None:
            return scores 
        
        loss, grads = 0, {}

        loss, dscores = softmax_loss(scores, y)

        loss += 0.5 * self.reg * (
            np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2)
        )

        d3, dW3, db3 = affine_backward(dscores, fc_cache)
        grads['W3'] = dW3 + self.reg * W3        
        grads['W3'] = db3

        d2, dW2, db2 = affine_relu_backward(d3, fc_relu_cache)
        grads['W2'] = dW2 + self.reg * W2        
        grads['b2'] = db2
        
        dX, dW1, db1 = conv_relu_maxpool_backward(d2, conv_cache)
        grads['W1'] = dW1 + self.reg * W1        
        grads['b1'] = db1


        return loss, grads 

    






     
    
    

    
    
 
        


#x = np.random.rand(4,3,5,5)
#fil = np.random.rand(3,3,3,3)
#b = np.random.rand(3,)
#
#dout = np.random.rand(4,3,5,5)
#
#model = Conv()
#out,cache = model.forward(x,fil,b)
#dx, dW, db = model.backward(dout , cache)
#
#print(f"shape of forward pass output: {out.shape}")
#print(f"shape of dx for backward pass: {dx.shape}")
#print(f"shape of dW for backward pass: {dW.shape}")
#print(f"shape of db for backward pass: {db.shape}")