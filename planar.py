import numpy as np
from sklearn.datasets import make_blobs

class DataGenerator:
    def planar_generator(self, mu, sigma, sample_size):
        # Planar equation
        theta =
        r = np.sin(4 * theta)

        # Convert polar equation to rectangular equation
        X1 = r * np.cos(theta)
        X2 = r * np.sin(theta)

        X = np.concatenate((X1, X2), axis=0)

        return X
