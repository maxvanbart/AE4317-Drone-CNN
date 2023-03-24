import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):

        super().__init__()
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)

        self.l1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.l2 = nn.Conv2d(8, 12, 3, 1, 1)
        self.l3 = nn.Conv2d(12, 20, 3, 1, 1)
        self.l4 = nn.Conv2d(20, 12, 1, 1, 0)
        self.l5 = nn.Conv2d(12, 20, 3, 1, 1)
        self.l6 = nn.Conv2d(20, 40, 3, 1, 1)
        self.l7 = nn.Conv2d(40, 20, 1, 1, 0)
        self.l8 = nn.Conv2d(20, 40, 3, 1, 1)
        self.l9 = nn.Conv2d(40, 100, 3, 1, 1)
        self.l10 = nn.Conv2d(100, 40, 1, 1, 0)
        self.l11 = nn.Conv2d(40, 100, 3, 1, 1)
        self.l12 = nn.Conv2d(100, 40, 1, 1, 0)
        self.l13 = nn.Conv2d(40, 100, 3, 1, 1)
        self.l14 = nn.Conv2d(100, 150, 3, 1, 1)
        self.l15 = nn.Conv2d(150, 40, 1, 1, 0)
        self.l16 = nn.Conv2d(40, 150, 3, 1, 1)
        self.l17 = nn.Conv2d(150, 40, 1, 1, 0)
        self.l18 = nn.Conv2d(40, 150, 3, 1, 1)

        self.layers = [self.max_pool,
                       self.l1,
                       self.relu,
                       self.max_pool,
                       self.l2,
                       self.relu,
                       self.max_pool,
                       self.l3,
                       self.relu,
                       self.l4,
                       self.relu,
                       self.l5,
                       self.relu,
                       self.max_pool,
                       self.l6,
                       self.relu,
                       self.l7,
                       self.relu,
                       self.l8,
                       self.relu,
                       self.max_pool,
                       self.l9,
                       self.relu,
                       self.l10,
                       self.relu,
                       self.l13,
                       self.relu,
                       self.max_pool,
                       self.l14,
                       self.relu]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
            print(x.shape)
        return x


def main():
    x = torch.zeros(1, 3, 520, 240)
    net = Net()
    y = net.forward(x)
    print(f"Final shape: {y.shape}")


if __name__ == "__main__":
    main()
