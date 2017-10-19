import numpy as np
from sklearn.datasets import make_blobs


class PlanarDataGenerator:

    def make_planar(self):
        self.mean = (0.25, 0.75)
        self.cov = [[0.15, 0], [0, 0.15]]
        self.size = 200
        X = np.random.multivariate_normal(self.mean, self.cov, self.size).reshape(-1, 1)
        y =
        # x, y = make_blobs(n_samples=400, n_features=1, cluster_std=0.15, centers=[(0.25,), (0.75,)])

        theta = X * np.pi * 2
        r = np.sin(4 * theta)

        x1 = r[:, 0] * np.cos(theta[:, 0])
        x2 = r[:, 1] * np.sin(theta[:, 1])

        # X = np.concatenate((x1, x2), axis=1).reshape(2, -1)

        print("The rose bloomed!")
        return X, y

    # def show_plot(self):
    #     # plt.subplot(121)
    #     # plt.scatter(th, r, c=y)
    #     # plt.subplot(122)
    #     # plt.scatter(x1, x2, c=y)
    #     # plt.show()
