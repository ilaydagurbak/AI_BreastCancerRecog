#!/usr/bin/env python3

import torch
import torchvision.models as models
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from torchvision.transforms import transforms
from constants import *
from model import Model

def main():

    alexnet = models.alexnet(pretrained=True)
    model = Model(alexnet,2)
    model.to(DEVICE)
	
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH,map_location= "cuda:0"))

    model.eval()


    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


    # input_image = Image.open("/home/Anyi/ATAI/BreastCancerRecognition/dataset/dataset/0/8863_idx5_x151_y1401_class0.png")
     input_image = Image.open("/home/Anyi/ATAI/BreastCancerRecognition/dataset/dataset/0/8863_idx5_x201_y401_class0.png")
    #input_image = Image.open("/home/Anyi/ATAI/BreastCancerRecognition/dataset/dataset/1/8959_idx5_x1101_y351_class1.png")
#    input_image = Image.open("/home/Anyi/ATAI/BreastCancerRecognition/dataset/dataset/1/8959_idx5_x1151_y251_class1.png")



    image_transform = transform(input_image)

    batch = torch.unsqueeze(image_transform, 0).to(DEVICE)
    out = model(batch)

    print(out)
    with open(LABELS_PATH) as f:
        classes = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    print(classes[index[0]], percentage[index[0]].item())

    _, indices = torch.sort(out, descending=True)
    [(classes[idx], percentage[idx].item()) for idx in indices[0][:]]
    y = classes[index[0]], percentage[index[0]].item()



if __name__ == '__main__':
    main()

