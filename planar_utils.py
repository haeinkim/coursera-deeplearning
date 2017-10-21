import numpy as np
import matplotlib.pyplot as plt


def load_planar_dataset(n_labels=2,
                        n_petals=4,
                        n_samples=200,
                        mean=(0.25, 0.75),
                        std=0.05):

    origin = np.random.uniform(low=0.0,
                               high=1.0,
                               size=n_labels * n_samples).reshape(1, -1)
    labels = np.empty([1, n_labels * n_samples])

    for label in range(n_labels):
        lbl = np.random.normal(loc=mean[label],
                               scale=std,
                               size=n_samples).reshape(1, -1)

        for idx, val in enumerate(lbl[0]):
            if val >= 0.5:
                lbl[0, idx] = 1
            else:
                lbl[0, idx] = 0

        labels[0, int(label * n_samples):int((label * n_samples) + n_samples)] \
            = lbl

    theta = np.sort(origin * np.pi * 2)
    r = 4 * np.sin(n_petals * theta)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    planar = np.r_[x, y]

    return planar, labels


def plot_decision_boundary(function, planar, labels, title):
    x_min, x_max = planar[0, :].min() - 1, planar[0, :].max() + 1
    y_min, y_max = planar[1, :].min() - 1, planar[1, :].max() + 1

    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    x = np.c_[xx.ravel(), yy.ravel()]
    z = function(x)
    z = z.reshape(xx.shape)

    _ = plt.contourf(xx, yy, z)
    _ = plt.scatter(planar[0, :], planar[1, :], c=labels, edgecolors='k', s=40)
    _ = plt.title(title)

    plt.show()
