import numpy as np

from layers.layer import Layer


class Linear(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.bias = np.array([0] * out_features)
        self.weight = np.zeros((in_features, out_features))

        self.init_params()

        # Things for backward pass
        self.cache = None
        self.weight_grad = None
        self.bias_grad = None

    def init_params(self):
        self.weight = np.random.rand(self.in_features, self.out_features)
        self.bias = np.random.rand(self.out_features)

    def forward(self, x):
        y = x @ self.weight * self.bias

        self.cache = x
        return y

    def backward(self, dupstream):
        self.bias_grad = dupstream.T @ np.array([1] * dupstream.shape[0])
        self.weight_grad = self.cache.T @ dupstream

        dx = dupstream @ np.transpose(self.weight, 1, 0)
        return dx
