import numpy as np
from src.utils import activationFunction, softMax

class NeuralNetwork:
    def __init__(self):
        # 1. "hidden layer", alkuun hard-koodattuna satunnaiset arvot, myöhemmin tulisi ladata oikeat arvot (treenaus ajojen jälkeen jne)
        # 784 neuronista -> 16 neuroniin
        self.W1=np.random.randn(16, 784) *0.01
        self.b1 = np.zeros((16,1))
        # 2. "hidden layer" 16 neuronista -> 16 neuroniin
        self.W2 = np.random.randn(16,16)*0.01
        self.b2 = np.zeros((16,1))
        # Output layer
        # 16 neuroista -> 10 neuroniin
        self.W3 = np.random.randn(10,16) * 0.01
        self.b3 = np.zeros((10,1))
    
    def forward_progpagation(self, X):
        # 1. kerroksen lineaarinen transformaatio + bias vektorin lisääminen
        Z1 = self.W1 @ X +self.b1
        A1 = activationFunction(Z1)
        # 2. kerroken linaarinen transformaatio + bias vektorin lisääminen
        Z2 = self.W2 @ A1 + self.b2
        A2 = activationFunction(Z2)
        # 3. kerroksen lineaarinen transformaatio + softmaxin käyttö
        Z3 = self.W3 @ A2 + self.b3
        A3 = softMax(Z3)
        return Z1, A1, Z2, A2, Z3, A3
