import numpy as np

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def relu(Z):
    A = np.max(Z, 0, Z)
    return A

def tanh(Z):
    A = np.tanh(Z)
    return A