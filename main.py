import numpy as np
import cv2

from layers.fullyConnected import Linear
from layers.relu import ReLU

def main(name):
    """Main function which does important things"""
    # # Take the imagename and turn it into the same format as the drone
    # image_name = name
    # im = cv2.imread(image_name)
    # YUV = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    # X = np.ndarray.flatten(YUV)
    # UYVY = []
    # for i in range(len(X) // 6):
    #     Y_ = [0, 0, 0, 0]
    #     X_ = X[6 * i:6 * (i + 1)]
    #     Y_[1] = int(X_[0])
    #     Y_[3] = int(X_[3])
    #     Y_[0] = int((int(X_[1]) + int(X_[4])) / 2)
    #     Y_[2] = int((int(X_[2]) + int(X_[5])) / 2)
    #     UYVY.append(Y_)
    # UYVY = np.ndarray.flatten(np.array(UYVY))

    l1 = Linear(3, 3)

    x = np.array([1, -2, 3])
    print(x)

    print(l1.weight, l1.bias)
    x = l1.forward(x)
    print(x)
    l2 = ReLU()
    print(l2.forward(x))



if __name__ == "__main__":
    main("242384873.jpg")
