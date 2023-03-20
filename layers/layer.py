

class Layer:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        return x

    def backward(self, dupstream):
        pass
