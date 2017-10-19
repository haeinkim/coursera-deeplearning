import numpy as np
from activations import tanh, sigmoid

np.random.seed(1)


class SingleLayerNet:
    """This implementation of the single layer neural network
    which is referred from the assignment "Planar data classification
    with one hidden layer
    """
    def __init__(self, X, hidden_size, Y, weight_init_std=0.01):
        """Define the structure and
        initialize the parameters of the two layer neural network"""
        self.n_x = X.shape[0]
        self.n_h = hidden_size
        self.n_y = Y.shape[0]

        self.parameters = {}
        self.parameters["W1"] = np.random.randn(self.n_x, self.n_h) \
                                * weight_init_std
        self.parameters["b1"] = np.zeros((self.n_h, 1))
        self.parameters["W2"] = np.random.randn(self.n_y, self.n_h) \
                                * weight_init_std
        self.parameters["b1"] = np.zeros((self.n_y, 1))

    def forward_propagation(self, X, parameters):
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]

        Z1 = np.dot(W1, X) + b1
        A1 = tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}

        return A2, cache

    def backward_propagation(self, parameters, cache, X, Y):
        m = X.shape[1]

        W1 = parameters["W1"]
        W2 = parameters["W2"]

        A1 = cache["A1"]
        A2 = cache["A2"]

        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.dot(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(W2.T, dZ2) * (1-np.power(A1, 2))
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads