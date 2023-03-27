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
        # print(f"### {self.img_name} ###")
        self.y = [0] * 352
        for thing in self.objects:
            i = int(thing[1]['x']) // 65
            j = int(thing[1]['y']) // 60
            x = thing[1]['x'] / h
            y = thing[1]['y'] / w

            pos = (4 * i + j) * 11

            # Warning: x and y are kinda flipped from their usual perspective but works if you turn image 90 degs
            self.y[pos] = x
            self.y[pos + 1] = y
            self.y[pos + 2] = thing[1]['width']
            self.y[pos + 3] = thing[1]['height']
            self.y[pos + 4] = 1
            self.y[pos + 5] = 0
            self.y[pos + 6] = 0
            self.y[pos + 7] = 0
            self.y[pos + 8] = 0
            self.y[pos + 9] = 0
            self.y[pos + 10] = 0

            match thing[0]:
                case 'pillar':
                    self.y[pos + 5] = 1
                case 'black_board':
                    self.y[pos + 6] = 1
                case 'white_board':
                    self.y[pos + 7] = 1
                case 'plant':
                    self.y[pos + 8] = 1
                case 'forest':
                    self.y[pos + 9] = 1
                case 'q_rcode':
                    self.y[pos + 10] = 1
        self.y = torch.Tensor(self.y)

    def to_input(self):
        # Take the imagename and turn it into the same format as the drone
        im = cv2.imread(f"captured_images/{self.img_name}")
        yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        yuv = np.swapaxes(yuv, 0, 2)
        yuv = torch.Tensor(np.swapaxes(yuv, 1, 2))
        yuv = yuv[None, :, :, :]

        self.x = yuv / 255


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = [nn.Conv2d(3, 192, 7, stride=2, padding=3),
                        nn.LeakyReLU(),
                        nn.MaxPool2d(2),

                        nn.Conv2d(192, 256, 3, padding=1),
                        nn.LeakyReLU(),
                        nn.MaxPool2d(2),

                        nn.Conv2d(256, 128, 1),
                        nn.LeakyReLU(),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.LeakyReLU(),
                        nn.Conv2d(256, 256, 1),
                        nn.LeakyReLU(),
                        nn.Conv2d(256, 512, 3, padding=1),
                        nn.MaxPool2d(2),

                        nn.Conv2d(512, 256, 1),
                        nn.LeakyReLU(),
                        nn.Conv2d(256, 512, 3, padding=1),
                        nn.LeakyReLU(),
                        nn.Conv2d(512, 256, 1),
                        nn.LeakyReLU(),
                        nn.Conv2d(256, 512, 3, padding=1),
                        nn.LeakyReLU(),
                        nn.Conv2d(512, 256, 1),
                        nn.LeakyReLU(),
                        nn.Conv2d(256, 512, 3, padding=1),
                        nn.LeakyReLU(),
                        nn.Conv2d(512, 256, 1),
                        nn.LeakyReLU(),
                        nn.Conv2d(256, 512, 3, padding=1),
                        nn.LeakyReLU(),
                        nn.Conv2d(512, 512, 1),
                        nn.LeakyReLU(),
                        nn.Conv2d(512, 1024, 3, padding=1),
                        nn.LeakyReLU(),
                        nn.MaxPool2d(2),

                        nn.Conv2d(1024, 512, 1),
                        nn.LeakyReLU(),
                        nn.Conv2d(512, 1024, 3, padding=1),
                        nn.LeakyReLU(),
                        nn.Conv2d(1024, 512, 1),
                        nn.LeakyReLU(),
                        nn.Conv2d(512, 1024, 3, padding=1),
                        nn.LeakyReLU(),
                        nn.Conv2d(1024, 1024, 3, padding=1),
                        nn.LeakyReLU(),
                        nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                        nn.LeakyReLU(),

                        nn.Conv2d(1024, 1024, 3, padding=1),
                        nn.LeakyReLU(),
                        nn.Conv2d(1024, 1024, 3, padding=1),
                        nn.LeakyReLU()]
        self.layers2 = [nn.Linear(32768, 4096),
                        nn.LeakyReLU(),
                        nn.Linear(4096, 11 * 8 * 4),
                        nn.LeakyReLU()]

    def forward(self, x):
        for layer in self.layers1:
            print(x.shape)
            x = layer.forward(x)

        x = torch.flatten(x)
        for layer in self.layers2:
            print(x.shape)
            x = layer.forward(x)
        return x


def main():
    with open('data.json') as file:
        data = json.load(file)

    print(f"Data length: {len(data)}.")
    # print(data[0]["Label"])

    images = []
    # for i in tqdm(range(len(data))):
    for i in tqdm(range(20)):
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
        image.to_output()

    x = torch.zeros(1, 3, 520, 240)
    net = Net()
    t0 = time()
    y = net.forward(x)
    t1 = time()
    dt = t1 - t0
    print(f"Forward pass took {dt} seconds.")
    print(f"Final shape: {y.shape}")


if __name__ == "__main__":
    main()
