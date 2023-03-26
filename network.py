import torch
import torch.nn as nn
from time import time
import json
import cv2
import numpy as np
from tqdm import tqdm

w = 240
h = 520


class Image:
    def __init__(self, name):
        self.img_name = name
        self.objects = []

        self.x = None
        self.y = None

    def to_output(self):
        pass

    def to_input(self):
        # Take the imagename and turn it into the same format as the drone
        im = cv2.imread(f"captured_images/{self.img_name}")
        yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        X = np.ndarray.flatten(yuv)
        uyvy = []
        for i in range(len(X) // 6):
            Y_ = [0, 0, 0, 0]
            X_ = X[6 * i:6 * (i + 1)]
            Y_[1] = int(X_[0])
            Y_[3] = int(X_[3])
            Y_[0] = int((int(X_[1]) + int(X_[4])) / 2)
            Y_[2] = int((int(X_[2]) + int(X_[5])) / 2)
            uyvy.append(Y_)
        uyvy = np.ndarray.flatten(np.array(uyvy))

        x = torch.zeros(1, 3, h, w)
        for i in range(int(w * h / 4)):
            pos = uyvy[4 * i: 4 * (i+1)]

            p1_c = i * 2
            p2_c = p1_c + 1
            p1_x = p1_c % w
            p1_y = p1_c // w
            p2_x = p2_c % w
            p2_y = p2_c // w

            # Store information in matrix (y, u, v)
            x[0, 0, p1_y, p1_x] = pos[1]
            x[0, 1, p1_y, p1_x] = pos[0]
            x[0, 2, p1_y, p1_x] = pos[2]
            x[0, 0, p2_y, p2_x] = pos[3]
            x[0, 1, p2_y, p2_x] = pos[0]
            x[0, 2, p2_y, p2_x] = pos[2]
        self.x = x


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

        # Not implemented in C layers
        self.l19 = nn.Conv2d(150, 200, 1, 1, 0)
        self.avg_pool = nn.AvgPool2d(200)

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
    with open('data.json') as file:
        data = json.load(file)

    print(f"Data length: {len(data)}.")
    # print(data[0]["Label"])

    images = []
    for i in tqdm(range(len(data))):
        dat = data[i]

        thing = Image(dat["External ID"])

        for item in dat["Label"]["objects"]:
            value = item["value"]
            bbox = item["bbox"]
            w = bbox["width"]
            h = bbox["height"]
            t = bbox["top"]
            l = bbox["left"]
            bbox["x"] = round(t + h / 2)
            bbox["y"] = round(l + w / 2)

            tup = (value, bbox)
            thing.objects.append(tup)
        images.append(thing)

    for image in tqdm(images):
        image.to_input()

    x = torch.zeros(1, 3, 520, 240)
    net = Net()
    t0 = time()
    y = net.forward(x)
    t1 = time()
    dt = t1-t0
    print(f"Forward pass took {dt} seconds.")
    print(f"Final shape: {y.shape}")


if __name__ == "__main__":
    main()
