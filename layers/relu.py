import numpy as np

from layers.layer import Layer


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.cache = x

        y = np.zeros(x.shape)
        for i in range(len(x)):
            y[i] = max(x[i], 0)
        return y

    def backward(self, dupstream):

        lst = []
        # For every value see on which side of the vertical axis it is in order to decide whether it has gradient 1 or 0
        for i in self.cache:
            lst.append([])
            for j in i:
                if j > 0:
                    lst[-1].append(1)
                else:
                    lst[-1].append(0)

        dx = dupstream * np.array(lst)
