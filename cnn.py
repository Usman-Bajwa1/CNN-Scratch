import numpy as np 

class Conv:

    def __init__(self, pad = 2, stride = 1):
 
        self.pad = pad
        self.stride  = stride 

    def __call__(self, x, W):
        
        pad = self.pad
        stride = self.stride

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
                        print(region_sum)
                        output[bs, oc, i, j] = region_sum
        
        cache = (x, W,)
        return output, cache
    
    def backward():
        pass


        


#fil = np.random.rand(3,3,3,3)
#x = np.random.rand(4,3,5,5)
#
#model = Conv(stride= 2)
#a = model(x,fil)
#print(a.shape)


