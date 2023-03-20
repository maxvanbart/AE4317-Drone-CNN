import numpy as np

from layers.layer import Layer


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = np.zeros(x.shape)
        for i in range(len(x)):
            y[i] = max(x[i], 0)
        return y

    def backward(self):
        pass
