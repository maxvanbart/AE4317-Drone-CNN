import torch
from tqdm import tqdm
import json

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
    for i in tqdm(range(1000, 1100)):
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

    y_pred = net(images[0])
    pass


if __name__ == "__main__":
    main()
