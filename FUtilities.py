# Imports here
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models, utils
import json
import scipy.io
from pprint import pprint
from collections import OrderedDict
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import argparse


# get the data.
def load_data():
    # paths for data.
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # transform data.
    train_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
    ])
    tv_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    # load the data.
    train_data = datasets.ImageFolder(train_dir,
                                transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir,
                                transform = tv_transforms)
    valid_data = datasets.ImageFolder(valid_dir,
                                transform = tv_transforms)

    # define dataloaders.
    trainloader = torch.utils.data.DataLoader(train_data,
                                        batch_size = 64,
                                        shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data,
                                        batch_size = 64)
    validloader = torch.utils.data.DataLoader(valid_data,
                                        batch_size = 64)
    class_to_idx = train_data.class_to_idx

    return trainloader, testloader, validloader, class_to_idx


# build model and set up network.
def build_model(hidden_layers=512, learning_rate=0.001, class_to_idx, power="gpu"):
    device = torch.device("cuda" if torch.cuda.is_available() and power="gpu" else "cpu")
    model = models.densenet161(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    classifier_input_size = model.classifier.in_features
    output_size = 102

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input_size, hidden_layers)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_layers, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    model.to(device)
    return model, criterion, optimizer


# train the model.
def train_model(model, learning_rate, criterion, optimizer, trainloader, validloader, power="gpu"):
    device = torch.device("cuda" if torch.cuda.is_available() and power="gpu" else "cpu")
    model.train()
    epochs = 10
    steps = 0
    running_loss = 0
    print_every = 50
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {test_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()


# save the checkpoint.
def save_checkpoint(checkpoint_path="densenet121_checkpoint.pth", hidden_layers=512, learning_rate=0.001, epochs=10):
    state = {
        "arch": "densenet121",
        "learning_rate": learning_rate,
        "hidden_layers": hidden_layers,
        "epochs": epochs,
        "state_dict": model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "class_to_idx" : model.class_to_idx
    }

# load the checkpoint.
def load_checkpoint(checkpoint_path="densenet121_checkpoint.pth"):
    state = torch.load(checkpoint_path)

    learning_rate = state["learning_rate"]
    class_to_idx = state["class_to_idx"]
    hidden_layers = state["hidden_layers"]

    model = build_model(hidden_layers, learning_rate, class_to_idx)

    model.load_state_dict(state["state_dict"])
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate=0.001)
    optimizer.load_state_dict(state["optimizer"])


# open & process data.
def process_image(image_path):
    import_image = Image.open(image_path)
    make_adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    adjusted_image = make_adjustments(import_image)
    return adjusted_image


# predict with processed picture
def predict(image_path="flowers/test/80/image_01983.jpg", model, topk=5, power="gpu"):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() and power="gpu" else "cpu")
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.float()
    
    with torch.no_grad():
        output = model.forward(image.to(device))
        top_prob, top_labels = torch.topk(output, topk)
        
        top_prob = nn.functional.softmax(output.data,dim=1)
        
    return top_prob.topk(topk)

def output(image_path="flowers/test/80/image01983.jpg", model):
    top_prob, top_classes = predict(check_image_path, model)
    top_classes = list(np.array(top_classes)[0])

    labels = []
    for class_idx in top_classes:
        string_label = list(class_to_idx.keys())[list(class_to_idx.values()).index(int(class_idx))]
        actual_label = cat_to_name[string_label]
        labels.append(actual_label)
                    
    a = list(np.array(top_prob[0]))
    b = labels

    print(f"Correct classification: {a[0]}")
    print(f"Correct prediction: {b[0]}")
