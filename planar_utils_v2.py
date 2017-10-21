import numpy as np


def load_planar_dataset(n_petals=4,
                        n_samples=200,
                        mean=[0.25, 0.75],
                        cov=[[0.2, 0], [0, 0.2]]):

    origin = np.random.multivariate_normal(mean,
                                           cov,
                                           n_samples).T.reshape(1, -1)
    label = np.c_[np.zeros((1, n_samples)), np.ones((1, n_samples))]

    theta = origin * np.pi * 2
    r = 4 * np.sin(n_petals * theta)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    planar = np.r_[x, y]

    print(planar.shape)
    print(label.shape)

    return planar, label
