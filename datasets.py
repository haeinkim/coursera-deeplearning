import numpy as np

z, label = make_blobs(n_samples=400, n_features=1, cluster_std=0.15,
                      centers=[(0.25,), (0.75,)])

theta = x * np.pi * 2
r = np.sin(n_petals * theta)

x = r * np.cos(theta)
y = r * np.sin(theta)

planar = np.c_[x, y].reshape(2, -1)
label = label.reshape(1, -1)




class PlanarDataGenerator:
    def __init__(self, n_labels=2, n_features=2, n_petals=4, mean=(0.25, 0.75),
                 scale=0.15, size=200):

        self.n_labels = n_labels
        self.n_features = n_features
        self.n_petals = n_petals
        self.mean = mean
        self.scale = scale
        self.size = size
        self.planar = np.empty([n_features, n_labels * size])
        self.label = np.c_[np.zeros((1, self.size)), np.ones((1, self.size))]

    def load_planar_dataset(self):
        for feature in range(self.n_features):
            z = np.random.normal(loc=self.mean[feature],
                                 scale=self.scale,
                                 size=self.size).reshape(1, -1)

            theta = z * np.pi * 2
            r = np.sin(self.n_petals * theta)

            x = r * np.cos(theta)
            y = r * np.sin(theta)

            self.planar[:, int(feature * self.size):
                           int(self.size + feature * self.size)] = np.r_[x, y]

        print(self.planar.shape)
        print(self.label.shape)

        print("The flower bloomed!")

        return self.planar, self.label
