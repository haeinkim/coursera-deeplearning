import numpy as np


def load_planar_dataset(n_labels=2,
                        n_features=2,
                        n_petals=4,
                        n_samples=200,
                        mean=(0.25, 0.75),
                        std=0.25):

    origin = np.empty([1, n_labels * n_samples])
    labels = np.empty([1, n_labels * n_samples])

    for feature in range(n_features):
        o = np.random.normal(mean[feature], std, n_samples).reshape(1, -1)

        origin[:, int(feature * n_samples):
             int((feature * n_samples) + n_samples)] = o

    for label in range(n_labels):
        l = np.random.normal(mean[label],
                             scale=0.5,
                             size=n_samples).reshape(1, -1)

        labels[0, int(label * n_samples):int((label * n_samples) + n_samples)] \
            = l

    for idx, val in enumerate(labels[0]):
        if val >= 0.5:
            labels[0, idx] = 1
        else:
            labels[0, idx] = 0

    theta = origin * np.pi * 2
    r = 4 * np.sin(n_petals * theta)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    planar = np.r_[x, y]

    print(planar.shape)
    print(labels.shape)

    return planar, labels
