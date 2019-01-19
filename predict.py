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

def data_import():
    data_dir = 'flower_data'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'



    # TODO: Define your transforms for the training, validation, and testing sets
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

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir,
                                     transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir,
                                    transform = tv_transforms)
    valid_data = datasets.ImageFolder(valid_dir,
                                     transform = tv_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data,
                                             batch_size = 64,
                                             shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data,
                                            batch_size = 64)
    validloader = torch.utils.data.DataLoader(valid_data,
                                             batch_size = 64)
    class_to_idx = train_data.class_to_idx
data_import()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)



def get_model():
    model = models.densenet161(pretrained=True)
    return model


# TODO: Build network and train model:
def build_model(hidden_layers, class_to_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
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
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    model.to(device)
    return model


# TODO.  Load the checkpoint.
state = torch.load("densenet121_checkpoint.pth")

learning_rate = state["learning_rate"]
class_to_idx = state["class_to_idx"]
hidden_layers = state["hidden_layers"]

model = build_model(hidden_layers, class_to_idx)

model.load_state_dict(state["state_dict"])
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
optimizer.load_state_dict(state["optimizer"])


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
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

process_image("flowers/test/1/image_06743.jpg")


def imshow(imp, title = None):
    # imshow for tensor.
    imp = imp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    imp = std * imp + mean
    imp = np.clip(imp, 0, 1)
    plt.imshow(imp)
    
    if title is not None:
        plt.title(title)


imshow(process_image("flowers/test/1/image_06743.jpg"), title=cat_to_name["1"])


model.class_to_idx =train_data.class_to_idx

ctx = model.class_to_idx

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.float()
    
    with torch.no_grad():
        output = model.forward(image.to(device))
        top_prob, top_labels = torch.topk(output, topk)
        
        top_prob = nn.functional.softmax(output.data,dim=1)
        
    return top_prob.topk(topk)


image_path = "flowers/test/80/image_01983.jpg"
predict(image_path, model, topk=5)


# TODO: Display an image along with the top 5 classes
check_image_path = "flowers/test/80/image_01983.jpg"

top_prob, top_classes = predict(check_image_path, model)
top_classes = list(np.array(top_classes)[0])

labels = []
for class_idx in top_classes:
    string_label = list(class_to_idx.keys())[list(class_to_idx.values()).index(int(class_idx))]
    actual_label = cat_to_name[string_label]
    labels.append(actual_label)
    
axs = imshow(process_image(check_image_path), title=cat_to_name["80"])
        
a = list(np.array(top_prob[0]))
b = labels
    
    
N=float(len(b))
fig,ax = plt.subplots(figsize=(8,3))
width = 0.5
tickLocations = np.arange(N)
ax.bar(tickLocations, a, width, linewidth=4.0, align = "center")
ax.set_xticks(ticks = tickLocations)
ax.set_xticklabels(b)
ax.set_xlim(min(tickLocations)-0.6,max(tickLocations)+0.6)
ax.set_yticks([0.2,0.4,0.6,0.8,1,1.2])
ax.set_ylim((0,1))
ax.yaxis.grid(True)

plt.show()
print(f"Correct classification: {a[0]}")
print(f"Correct prediction: {b[0]}")
# How do you turn the plot sideways?

