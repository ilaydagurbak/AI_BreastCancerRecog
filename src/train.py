
import copy
from multiprocessing.spawn import freeze_support
import torch
import torch.nn as nn
import requests
import tqdm as tqdm
from tqdm import tqdm
import torch.optim as optim 
import adabound as  ab
import torchvision.models as models
from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
from model import Model
from constants import *
def main() :

    alexnet = models.alexnet(pretrained=True)
    model = Model(alexnet, 2)
    model.to(DEVICE)

    epoch_arr = np.arange(EPOCH)
    points_arr = np.empty(0)
    train_pts = np.empty(0)
    print(torch.cuda.device_count()) 
    dataset_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = ImageFolder(DATASET_PATH, transform=dataset_transform)
    test_len = int(len(dataset) / 3)
    train_len = int(len(dataset) - test_len)

    train, test = data.random_split(dataset, [train_len, test_len])
    train_set = data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NO_WORKERS)
    test_set = data.DataLoader(test, batch_size=512, shuffle=False, num_workers=NO_WORKERS)

    for p in model.features.parameters() :
        p.requires_grad = False

    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.00001)


    best_model = copy.deepcopy(alexnet.state_dict())
    best_acc = 0.0
    epoch_no_improve = 0
    best_train_error = 1.5
    counter = 0

    for epoch in tqdm(range(EPOCH),desc="epochs"):
        model.train()

        # counts number of epochs that are not improving
        training_error = 0.0
        correct = 0
        correct_train = 0
        total_train = 0
        total_test = 0
        val_loss = 0.0
        max_wait_epoch = 5

        for i, batch in enumerate(tqdm(train_set,desc="training progress")):

            optimizer.zero_grad()

            running_corrects = 0

            image_batch, label_batch = batch
            image_batch = image_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)

            predicted_labels = model(image_batch)
            predicted_labels = predicted_labels.to(DEVICE)
            _, predictions = torch.max(predicted_labels, 1)

            loss = loss_func(predicted_labels, label_batch)
            training_error +=  loss.item()


           # training_error = training_error + loss.item()
            loss.backward()

            optimizer.step()
            running_corrects += torch.sum(predictions.to(DEVICE) == label_batch)

            total_train += label_batch.size(0)
            correct_train += (predictions == label_batch).sum().item()
        print("EPOCH: ", epoch, "total_train: ", total_train, "total correct: ", correct_train)
        train_loss = training_error/64
        train_pts = np.append(train_pts, train_loss)

        print('Accuracy of training on the all the images : %d %%' % (
                100 * correct_train / total_train))

        model.eval()
        for i, d in enumerate(tqdm(test_set, desc="testing progress")) :
           #	 print("iteration: ", i, "/", len(test_set))
            test_image, test_label = d
            test_image = test_image.to(DEVICE)
            test_label = test_label.to(DEVICE)

            output = model(test_image)
            loss = loss_func(output, test_label)

            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_test += test_label.size(0)
            correct += (predicted == test_label).sum().item()
        print("correct: ", correct, " total: ", total_test)

        print('Accuracy of the network on the all the images test images: %d %%' % (
                100 * correct / total_test))
        test_loss =  val_loss/512
        print(test_loss)
        points_arr = np.append(points_arr, test_loss)

        if test_loss < best_train_error :
            best_model = copy.deepcopy(model.state_dict())
            best_train_error = test_loss
            print("BEST TRAIN ERROR: ", best_train_error)
            counter = 0
        else :
            counter += 1
            print("EPOCHS WITHOUT IMPROVING: ", counter)

            if counter >= max_wait_epoch :
                plt.savefig("experiment2_lr00001.png")
                print("STOPPED EARLY! BEST MODEL FOUND")
                return torch.save(best_model, TRAINED_MODEL_PATH)


    torch.save(best_model, TRAINED_MODEL_PATH)

    avg_train_loss = loss / len(train_set)
    avg_test_loss = val_loss / len(test_set)

    plt.plot(epoch_arr,points_arr)
    print(train_pts)
    plt.savefig("experiment2_lr00001.png")
    plt.show()


    print("avg train: ", avg_train_loss, "avg test: ", avg_test_loss)


if __name__ == '__main__' :
    freeze_support()
    main()
    exit()

