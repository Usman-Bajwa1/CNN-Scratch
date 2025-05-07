import numpy as np 
from cnn import ThreeLayerConvNet


def main():
        
    N, I_C, H, W = 50, 3, 32, 32
    
    X = np.random.rand(N, I_C, H, W)
    y = np.random.randint(10, size=N)
    
    model = ThreeLayerConvNet()

    loss, grads = model.loss(X, y) 

    print(f"Initial loss: {loss}")

    model.reg = 0.5
    loss, grads = model.loss(X, y)
    print(f"Loss with reg: {loss}")

main()
