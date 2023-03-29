import torch
from tqdm import tqdm
import json
import numpy as np

from network import Net, Image


def main():
    print("Hello world!")
    net = Net()
    net.load_state_dict(torch.load("model/yolo1"))
    net.eval()

    # Load labelbox json file for hand drawn boxes
    with open('data.json') as file:
        data = json.load(file)

    print(f"Data length: {len(data)}.")

    images = []
    # for i in tqdm(range(len(data))):
    # for j in range(10):
    for i in tqdm(range(1600, 1700)):
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

    n = 90
    y_pred = net(images[n].x).cpu().detach().numpy()
    pred_mat = []
    print(f"### {images[n].img_name} ###")
    for i in range(32):
        row = y_pred[11*i:11*i+11]
        row_dat = row[:5]
        row_prob = row[5:]
        row_prob = np.exp(row_prob) / sum(np.exp(row_prob))
        lst = list(row_dat) + list(row_prob)
        if max(list(row_prob)) > 0.17:
            print(f"Found object in window {i} with following probabilities: ")
            print(list(row_prob))

        pred_mat.append(lst)
    pred_mat = np.array(pred_mat)
    pass


if __name__ == "__main__":
    main()
