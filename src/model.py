import numpy as np
from src.utils import relu, softMax

class NeuralNetwork:
    def __init__(self):
        # Hidden layer 1, initally hard coded the starting values
        # maps 784 neurons to 16 neurons
        self.W1=np.random.randn(16, 784) *0.01
        self.b1 = np.zeros((16,1))
        # Hidden layer 2
        self.W2 = np.random.randn(16,16)*0.01
        self.b2 = np.zeros((16,1))
        # Output layer maps from 16 neurons to 10 neurons
        self.W3 = np.random.randn(10,16) * 0.01
        self.b3 = np.zeros((10,1))
    
    def forward_progpagation(self, X):
        # Hidden layer 1 linear transformation + bias vector
        Z1 = self.W1 @ X +self.b1
        A1 = relu(Z1)
        # Hidden layer 2 + bias vector
        Z2 = self.W2 @ A1 + self.b2
        A2 = relu(Z2)
        # Output layer + softmax
        Z3 = self.W3 @ A2 + self.b3
        A3 = softMax(Z3)
        return Z1, A1, Z2, A2, Z3, A3
    
    def backward_propagation(self, X, Y, Z1, A1, Z2, A2, A3):
        m = X.shape[1]
        
        # Output Layer
        dZ3 = A3 - Y 
        dW3 = (1/m) * (dZ3 @ A2.T)
        db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)
        
        # Hidden Layer 2
        dZ2 = (self.W3.T @ dZ3) * (Z2 > 0)
        dW2 = (1/m) * (dZ2 @ A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # Hidden Layer 1
        dZ1 = (self.W2.T @ dZ2) * (Z1 > 0)
        dW1 = (1/m) * (dZ1 @ X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        
        return dW1, db1, dW2, db2, dW3, db3
    
    def compute_loss(self, A3, Y):
        m = Y.shape[1]
        # tiny epsilon to avoid log(0)
        loss = -1/m * np.sum(Y*np.log(A3+1e-15))
        return loss


    # Updates the models hyperparameters
    def step(self, dW1, db1, dW2, db2, dW3, db3, lr):
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3

