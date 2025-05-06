import numpy as np 


def sgd(w, b, db, dw, config = None):

    if config is None:
        config = {}
    lr = config.setdefault("learning_rate",1e-2)

    w -= lr*dw
    b -= lr*db 
    
    return w,b, config

def sgd_momentum(w, b, db, dw, config = None):

    if config is None:
        config = {}
    
    lr = config.setdefault("learning_rate", 1e-2)
    mo = config.setdefault("momentum",0.9)
    velocity_w = config.setdefault("velocity_w",np.zeros_like(w))
    velocity_b = config.setdefault("velocity_b",np.zeros_like(b))

    velocity_w = mo * velocity_w + (1 - mo) * dw
    velocity_b = mo * velocity_b + (1 - mo) * db

    w -= lr * velocity_w
    b -= lr * velocity_b

    config["velocity_w"] =  velocity_w
    config["velocity_b"] =  velocity_b    
    return w, b, config

def rms_prop(w, b, db, dw, config = None):

    if config is None:
        config = {}

    lr = config.setdefault("learning_rate", 1e-2)
    beta = config.setdefault("decay_rate", 0.99)
    ep = config.setdefault("epsilon",1e-8)
    cache_w = config.setdefault("cache_w", np.zeros_like(w))
    cache_b = config.setdefault("cache_b", np.zeros_like(b))

    cache_w = beta * cache_w + (1 - beta) * (dw ** 2)
    cache_b = beta * cache_b + (1 - beta) * (db ** 2)

    w -= lr * dw / (np.sqrt(cache_w) + ep)
    b -= lr * db / (np.sqrt(cache_b) + ep)
    config["cache_w"] = cache_w
    config["cache_b"] = cache_b    

    return w, b, config

def adam(w, b, db, dw, config = None):

    if config is None:
        config = {}

    lr = config.setdefault("learning_rate",1e-2)
    mo = config.setdefault("momentum", 0.9)
    beta = config.setdefault("decay_rate", 0.99)
    ep = config.setdefault("epsilon",1e-8)
    t = config.setdefault("t", 0)

    velocity_w = config.setdefault("velocity_w", np.zeros_like(w))
    velocity_b = config.setdefault("velocity_b", np.zeros_like(b))

    cache_w = config.setdefault("cache_w",np.zeros_like(w))
    cache_b = config.setdefault("cache_b",np.zeros_like(b))

    t += 1

    velocity_w = mo * velocity_w + (1 - mo) * dw
    velocity_b = mo * velocity_b + (1 - mo) * db

    cache_w = beta * cache_w + (1 - beta) * (dw ** 2)    
    cache_b = beta * cache_b + (1 - beta) * (db ** 2)

    mt_w = velocity_w / (1 - mo ** t)
    mt_b = velocity_b / (1 - mo ** t)

    vt_w = cache_w / (1 - beta ** t)
    vt_b = cache_b / (1 - beta ** t)

    w -= lr * mt_w / (np.sqrt(vt_w) + ep)
    b -= lr * mt_b / (np.sqrt(vt_b) + ep)

    config["t"] = t
    config["velocity_w"] = velocity_w
    config["velocity_b"] = velocity_b
    config["cache_w"] = cache_w
    config["cache_b"] = cache_b

    return  w, b, config










