import torch
from tqdm import tqdm
import json
import numpy as np
import pandas as pd

from network import Net, Image


def main():
    print("Hello world!")
    # Initialize network based on trained parameters from the network.py file
    net = Net()
    net.load_state_dict(torch.load("model/yolo1"))
    net.eval()

    # Load labelbox json file for hand drawn boxes
    with open('data.json') as file:
        data = json.load(file)
    # Show length of the labelbox data
    print(f"Data length: {len(data)}.")

    # Create an empty list to store the initialized image objects
    images = []
    # Load the images in the desired range for verification
    for i in tqdm(range(2000, 2100)):
        # Extract the indexed json format things
        dat = data[i]

        # Initialize an image class object using the name of the image
        thing = Image(dat["External ID"])

        # For all obstacles stored in the labeled data calculate the parameters which the yolo paper desires
        # as outputs
        for item in dat["Label"]["objects"]:
            value = item["value"]
            bbox = item["bbox"]
            w = bbox["width"]
            h = bbox["height"]
            t = bbox["top"]
            l = bbox["left"]
            bbox["x"] = round(t + h / 2)
            bbox["y"] = round(l + w / 2)
            # Store the data on the bounding box of the object and the object type in a tuple in the list containing
            # the objects.
            tup = (value, bbox)
            thing.objects.append(tup)
        # Append the initialized image class to the images container
        images.append(thing)

    # Generate all the image input and output data
    for image in tqdm(images):
        image.to_input()
        image.to_output()

    # Print information on image n in the images container to check model accuracy
    n = 90
    # Get the final result back from the gpu if it is stored there such that it can be displayed
    y_pred = net(images[n].x).cpu().detach().numpy()
    # Make a matrix to store the results for the different grid windows
    pred_mat = []
    # Print the image name such that the associated image can be easily found
    print(f"### {images[n].img_name} ###")
    # For every grid cell get the prediction row
    for i in range(32):
        # Get the correct elements from the output array
        row = y_pred[11*i:11*i+11]
        # Separate the class probabilities from the other information
        row_dat = row[:5]
        row_prob = row[5:]
        # Apply a softmax filter to the class probabilities
        row_prob = np.exp(row_prob) / sum(np.exp(row_prob))
        # Add the two rows back together for the final matrix
        lst = list(row_dat) + list(row_prob)
        # Print the rows which are likely to contain an object
        if row_dat[4] > 0:
            # Print the index of the grid cell where an object was detected and the row of probabilities
            print(f"Found object in window {i} with following probabilities: ")
            #columns = ["pillar", "black board", "white board", "plant", "forest", "qr code"]
            #df = pd.DataFrame(np.array(row_prob), columns=columns)
            print(row_prob)

        pred_mat.append(lst)
    # Make a full 2d prediction such that this can be accessed in the debugger
    pred_mat = np.array(pred_mat)
    pass


if __name__ == "__main__":
    main()
