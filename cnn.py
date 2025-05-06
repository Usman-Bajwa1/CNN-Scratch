import numpy as np 

class Conv:

    def __init__(self, conv_param = {"stride": 1, "pad": 1}):
 
        self.conv_param = conv_param
     
    
    

    
    
 
        


x = np.random.rand(4,3,5,5)
fil = np.random.rand(3,3,3,3)
b = np.random.rand(3,)

dout = np.random.rand(4,3,5,5)

model = Conv()
out,cache = model.forward(x,fil,b)
dx, dW, db = model.backward(dout , cache)

print(f"shape of forward pass output: {out.shape}")
print(f"shape of dx for backward pass: {dx.shape}")
print(f"shape of dW for backward pass: {dW.shape}")
print(f"shape of db for backward pass: {db.shape}")