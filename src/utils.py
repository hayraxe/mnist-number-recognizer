import numpy as np

# activation function
def relu(Z):
    return np.maximum(0,Z)
    

def softMax(Z):
    # z is (10,m)
    # shift  Z by subtractring the max for stability
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)