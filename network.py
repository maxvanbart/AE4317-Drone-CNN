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

# run on GPU if available
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
        # find centroid of each object class
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

            if thing[0] == 'pillar':
                self.y[pos + 5] = 1
            elif thing[0] == 'black_board':
                self.y[pos + 6] = 1
            elif thing[0] == 'white_board':
                self.y[pos + 7] = 1
            elif thing[0] == 'plant':
                self.y[pos + 8] = 1
            elif thing[0] == 'forest':
                self.y[pos + 9] = 1
            elif thing[0] == 'q_rcode':
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
        #define network layers
        self.mp0 = nn.MaxPool2d(2)
        self.c1 = nn.Conv2d(3, 100, 7, stride=2, padding=3, device=device)
        self.r1 = nn.LeakyReLU()
        self.mp1 = nn.MaxPool2d(2)

        self.c2 = nn.Conv2d(100, 100, 3, padding=1, device=device)
        self.r2 = nn.LeakyReLU()
        self.mp2 = nn.MaxPool2d(2)

        self.c3 = nn.Conv2d(100, 64, 1, device=device)
        self.r3 = nn.ReLU()
        self.c4 = nn.Conv2d(64, 100, 3, padding=1, device=device)
        self.r4 = nn.ReLU()
        self.c6 = nn.Conv2d(100, 150, 3, padding=1, device=device)
        self.r6 = nn.LeakyReLU()
        self.mp3 = nn.MaxPool2d(2)

        self.c7 = nn.Conv2d(150, 100, 1, device=device)
        self.r7 = nn.ReLU()
        self.c8 = nn.Conv2d(100, 150, 3, padding=1, device=device)
        self.r8 = nn.ReLU()
        self.c15 = nn.Conv2d(150, 150, 1, device=device)
        self.r15 = nn.LeakyReLU()
        self.c16 = nn.Conv2d(150, 250, 3, padding=1, device=device)
        self.r16 = nn.ReLU()
        self.mp4 = nn.MaxPool2d(2)

        self.c17 = nn.Conv2d(250, 150, 1, device=device)
        self.r17 = nn.LeakyReLU()
        self.c18 = nn.Conv2d(150, 250, 3, padding=1, device=device)
        self.r18 = nn.ReLU()
        self.c22 = nn.Conv2d(250, 250, 3, stride=2, padding=1, device=device)
        self.r22 = nn.ReLU()
        self.c24 = nn.Conv2d(250, 250, 3, padding=1, device=device)
        self.r24 = nn.LeakyReLU()

        #construct network architecture
        self.layers1 = [self.mp0,
                        self.c1,
                        self.r1,
                        self.mp1,

                        self.c2,
                        self.r2,
                        self.mp2,

                        self.c3,
                        self.r3,
                        self.c4,
                        self.r4,
                        self.c6,
                        self.r6,
                        self.mp3,

                        self.c7,
                        self.r7,
                        self.c8,
                        self.r8,
                        self.c15,
                        self.r15,
                        self.c16,
                        self.r16,
                        self.mp4,

                        self.c17,
                        self.r17,
                        self.c18,
                        self.r18,
                        self.c22,
                        self.r22,

                        self.c24,
                        self.r24]

        #define fully connected layers
        self.l1 = nn.Linear(2000, 4096, device=device)
        self.r25 = nn.LeakyReLU()
        self.l2 = nn.Linear(4096, 11 * 8 * 4, device=device)
        self.r26 = nn.LeakyReLU()

        #define fully connected architecture
        self.layers2 = [self.l1,
                        self.r25,
                        self.l2,
                        self.r26]

    def forward(self, x):
        # forward pass of cnn architecture
        for layer in self.layers1:
            # print(x.shape)
            x = layer.forward(x)

        x = torch.flatten(x)
        for layer in self.layers2:
            # print(x.shape)
            x = layer.forward(x)
        return x

    #reset network parameters after pass
    def init(self):
        for layer in self.layers1 + self.layers2:
            layer.reset_parameters()

    def export_weights(self):
        for i, layer in enumerate(self.layers1):
            # Save layer weights to csv
            if hasattr(layer, 'weight'):
                print(f"Convolutional layer {i} has weights.")
                file = ""
                weight = layer.weight.detach().numpy()

                # Reduce dimension of kernal to three
                for j in range(weight.shape[0]):
                    kernel = weight[j, :, :, :]
                    kernel = np.ndarray.flatten(kernel)
                    kernel_str = ""
                    for item in kernel:
                        kernel_str += f"{item},"
                    kernel_str.rstrip(',')
                    kernel_str += '\n'
                    file += kernel_str

                with open(f"weights/weight{i}.csv", 'w') as f:
                    f.write(file)

            # Save biases to csv
            if hasattr(layer, 'bias'):
                print(f"Convolutional layer {i} has biases")
                bias = layer.bias.detach().numpy()
                np.savetxt(f"weights/bias{i}.csv", bias, delimiter=",")
                # tensor_to_dat(bias, f"weights/bias{i}.csv")

        for i, layer in enumerate(self.layers2):
            # Save linear weights to csv
            if hasattr(layer, 'weight'):
                print(f"Linear layer {i + len(self.layers1)} has weights")
                weight = layer.weight.detach().numpy()
                np.savetxt(f"weights/weight{i + len(self.layers1)}.csv", np.ndarray.flatten(weight), delimiter=",")
                # tensor_to_dat(bias, f"weights/bias{i}.csv")

            # Save biases to csv
            if hasattr(layer, 'bias'):
                print(f"Linear layer {i + len(self.layers1)} has biases")
                bias = layer.bias.detach().numpy()
                np.savetxt(f"weights/bias{i + len(self.layers1)}.csv", bias, delimiter=",")
                # tensor_to_dat(bias, f"weights/bias{i}.csv")


def main():
    # Load labelbox json file for hand drawn boxes
    with open('data.json') as file:
        data = json.load(file)

    print(f"Data length: {len(data)}.")

    images = []

    for i in tqdm(range(1)):
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

            # object type and bounding box dimensions
            tup = (value, bbox)
            thing.objects.append(tup)
        # add details to images
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

    # Optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=0.00005)

    # Save the model parameters to a data file
    # net.export_weights()
    # torch.save(net.state_dict(), "weights/model.dat")

    losses = []

    epochs = 10
    for epoch in range(epochs):
        print(f'Starting epoch {epoch}.')
        running_loss = 0.0
        for j in range(5):
            # Clear images from memory
            images = []
            # Load new images into memory
            for i in tqdm(range(1000*j, 1000*j+1000, 10)):
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

            # Make image data
            for image in tqdm(images):
                image.to_input()
                image.to_output()

            # Train network with loaded images
            for i, data_ in tqdm(enumerate(images, 0)):
                x = data_.x
                y_true = data_.y

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
    # save trained data to model
    torch.save(net.state_dict(), "model/yolo1")
    plt.plot(range(epochs), losses)
    plt.show()

# run and time
if __name__ == "__main__":
    t0 = time()
    main()
    t1 = time()
    dt = t1 - t0
    print(f"Total run took {dt} seconds.")