import cv2
import numpy as np
import matplotlib.pyplot as plt

image_name = '242384873.jpg'
im = cv2.imread(image_name)
YUV = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
X = np.ndarray.flatten(YUV)
UYVY= []
for i in range(len(X)//6):
    Y_ = [0, 0, 0, 0]
    X_ = X[6*i:6*(i+1)]
    Y_[1] = int(X_[0])
    Y_[3] = int(X_[3])
    Y_[0] = int((int(X_[1]) + int(X_[4]))/2)
    Y_[2] = int((int(X_[2]) + int(X_[5]))/2)
    UYVY.append(Y_)

UYVY = np.ndarray.flatten(np.array(UYVY))
with np.printoptions(threshold=np.inf):
        print(X)
        print(UYVY)
pass