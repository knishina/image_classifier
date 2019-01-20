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
import FUtilities


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", dest="gpu", action="store", help="Default is gpu", default="gpu")
parser.add_argument("image_path", action="store", nargs="*", help="Default is flowers/test/80/image_01983.jpg", default="../aipnd-project/flowers/test/80/image_01983.jpg")
parser.add_argument("checkpoint_path", action="store", nargs='*', help="Default is checkpoint.pth", default="checkpoint.pth")
parser.add_argument("--topk", action="store", dest="topk", type=int, help="Default is 5", default=5)
parser.add_argument("--cat_to_name", action="store", dest="cat_to_name", help="Default is cat_to_name", default="cat_to_name")
parser.add_argument("--hidden_layers", dest="hidden_layers", action="store", type=int, help="Default is 512", default=512)
parser.add_argument("--learning_rate", dest= "learning_rate", action="store", type=int, help="Default is 0.001", default=0.001)
parser.add_argument("--epochs", dest="epochs", action="store", type=int, help="Default is 10", default=10)


par = parser.parse_args()
power = par.gpu
path = par.checkpoint_path
image_path = par.image_path
topk = par.topk
cat_to_name = par.cat_to_name
learning_rate = par.learning_rate
hidden_layers = par.hidden_layers
epochs = par.epochs


trainloader, testloader, validloader, class_to_idx = FUtilities.load_data()
model = FUtilities.load_checkpoint(path)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

FUtilities.predict(image_path, model, topk, power)
FUtilities.output(image_path, topk, model, power, class_to_idx, cat_to_name)
print ("doned.")