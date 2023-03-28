import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from time import time
import json
import cv2
import numpy as np
from tqdm import tqdm

w = 240
h = 520

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

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
        self.y = torch.Tensor(self.y).to(device)

    def to_input(self):
        # Take the imagename and turn it into the same format as the drone
        im = cv2.imread(f"captured_images/{self.img_name}")
        yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        yuv = np.swapaxes(yuv, 0, 2)
        yuv = torch.Tensor(np.swapaxes(yuv, 1, 2))
        yuv = yuv[None, :, :, :]

        self.x = (yuv / 255).to(device)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 192, 7, stride=2, padding=3, device=device)
        self.r1 = nn.LeakyReLU()
        self.mp1 = nn.MaxPool2d(2)

        self.c2 = nn.Conv2d(192, 256, 3, padding=1, device=device)
        self.r2 = nn.LeakyReLU()
        self.mp2 = nn.MaxPool2d(2)

        self.c3 = nn.Conv2d(256, 128, 1, device=device)
        self.r3 = nn.LeakyReLU()
        self.c4 = nn.Conv2d(128, 256, 3, padding=1, device=device)
        self.r4 = nn.LeakyReLU()
        self.c5 = nn.Conv2d(256, 256, 1, device=device)
        self.r5 = nn.LeakyReLU()
        self.c6 = nn.Conv2d(256, 512, 3, padding=1, device=device)
        self.r6 = nn.LeakyReLU()
        self.mp3 = nn.MaxPool2d(2)

        self.c7 = nn.Conv2d(512, 256, 1, device=device)
        self.r7 = nn.LeakyReLU()
        self.c8 = nn.Conv2d(256, 512, 3, padding=1, device=device)
        self.r8 = nn.LeakyReLU()
        self.c9 = nn.Conv2d(512, 256, 1, device=device)
        self.r9 = nn.LeakyReLU()
        self.c10 = nn.Conv2d(256, 512, 3, padding=1, device=device)
        self.r10 = nn.LeakyReLU()
        self.c11 = nn.Conv2d(512, 256, 1, device=device)
        self.r11 = nn.LeakyReLU()
        self.c12 = nn.Conv2d(256, 512, 3, padding=1, device=device)
        self.r12 = nn.LeakyReLU()
        self.c13 = nn.Conv2d(512, 256, 1, device=device)
        self.r13 = nn.LeakyReLU()
        self.c14 = nn.Conv2d(256, 512, 3, padding=1, device=device)
        self.r14 = nn.LeakyReLU()
        self.c15 = nn.Conv2d(512, 512, 1, device=device)
        self.r15 = nn.LeakyReLU()
        self.c16 = nn.Conv2d(512, 1024, 3, padding=1, device=device)
        self.r16 = nn.LeakyReLU()
        self.mp4 = nn.MaxPool2d(2)

        self.c17 = nn.Conv2d(1024, 512, 1, device=device)
        self.r17 = nn.LeakyReLU()
        self.c18 = nn.Conv2d(512, 1024, 3, padding=1, device=device)
        self.r18 = nn.LeakyReLU()
        self.c19 = nn.Conv2d(1024, 512, 1, device=device)
        self.r19 = nn.LeakyReLU()
        self.c20 = nn.Conv2d(512, 1024, 3, padding=1, device=device)
        self.r20 = nn.LeakyReLU()
        self.c21 = nn.Conv2d(1024, 1024, 3, padding=1, device=device)
        self.r21 = nn.LeakyReLU()
        self.c22 = nn.Conv2d(1024, 1024, 3, stride=2, padding=1, device=device)
        self.r22 = nn.LeakyReLU()

        self.c23 = nn.Conv2d(1024, 1024, 3, padding=1, device=device)
        self.r23 = nn.LeakyReLU()
        self.c24 = nn.Conv2d(1024, 1024, 3, padding=1, device=device)
        self.r24 = nn.LeakyReLU()

        self.layers1 = [self.c1,
                        self.r1,
                        self.mp1,

                        self.c2,
                        self.r2,
                        self.mp2,

                        self.c3,
                        self.r3,
                        self.c4,
                        self.r4,
                        self.c5,
                        self.r5,
                        self.c6,
                        self.r6,
                        self.mp3,

                        self.c7,
                        self.r7,
                        self.c8,
                        self.r8,
                        self.c9,
                        self.r9,
                        self.c10,
                        self.r10,
                        self.c11,
                        self.r11,
                        self.c12,
                        self.r12,
                        self.c13,
                        self.r13,
                        self.c14,
                        self.r14,
                        self.c15,
                        self.r15,
                        self.c16,
                        self.r16,
                        self.mp4,

                        self.c17,
                        self.r17,
                        self.c18,
                        self.r18,
                        self.c19,
                        self.r19,
                        self.c20,
                        self.r20,
                        self.c21,
                        self.r21,
                        self.c22,
                        self.r22,

                        self.c23,
                        self.r23,
                        self.c24,
                        self.r24]

        self.l1 = nn.Linear(32768, 4096, device=device)
        self.r25 = nn.LeakyReLU()
        self.l2 = nn.Linear(4096, 11 * 8 * 4, device=device)
        self.r26 = nn.LeakyReLU()

        self.layers2 = [self.l1,
                        self.r25,
                        self.l2,
                        self.r26]

    def forward(self, x):
        for layer in self.layers1:
            # print(x.shape)
            x = layer.forward(x)

        x = torch.flatten(x)
        for layer in self.layers2:
            # print(x.shape)
            x = layer.forward(x)
        return x

    def init(self):
        for layer in self.layers1 + self.layers2:
            layer.reset_parameters()

    # def backward(self, dy):
    #


def main():
    # Load labelbox json file for hand drawn boxes
    with open('data.json') as file:
        data = json.load(file)

    print(f"Data length: {len(data)}.")

    images = []
    # for i in tqdm(range(len(data))):
    #for j in range(10):
    for i in tqdm(range(100)):
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

    # Generate all the image input and output data
    for image in tqdm(images):
        image.to_input()
        image.to_output()

    # Make a net to train based on the yolov1 architecture
    x = torch.zeros(1, 3, 520, 240).to(device)
    net = Net()
    t0 = time()
    y = net.forward(x)
    t1 = time()
    dt = t1 - t0
    print(f"Forward pass took {dt} seconds.")
    print(f"Final shape: {y.shape}")

    params = list(net.parameters())
    # print(params)
    # print(len(params))
    # print(params[0].size())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.00005, momentum=0.9)

    losses = []

    epochs = 50
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in tqdm(enumerate(images, 0)):
            x = data.x
            y_true = data.y

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            y = net(x)
            loss = criterion(y, y_true)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()

        losses.append(running_loss)
        print(running_loss, epoch)

    print(losses)
    plt.plot(range(epochs), losses)
    plt.show()


if __name__ == "__main__":
    main()
