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
import FUtilities


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", dest="gpu", action="store", help="Default is gpu", default="gpu")
parser.add_argument("--checkpoint_path", dest="checkpoint_path", action="store", help="Default is densenet121_checkpoint.pth", default="./densenet121_checkpoint.pth")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument("--hidden_layers", dest="hidden_layers", action="store", type=int, help="Default is 512", default=512)
parser.add_argument("--learning_rate", dest= "learning_rate", action="store", type=int, help="Default is 0.001", default=0.001)
parser.add_argument("--epochs", dest="epochs", action="store", type=int, help="Default is 10", default=10)

par = parser.parse_args()
checkpoint_path = par.checkpoint_path
learning_rate = par.learning_rate
hidden_layers = par.hidden_layers
power = par.gpu
epochs = par.epochs
path = par.save_dir



trainloader, validloader, testloader, class_to_idx = FUtilities.load_data()
model, criterion, optimizer = FUtilities.build_model(hidden_layers, learning_rate, class_to_idx, power)
FUtilities.train_model(model, learning_rate, criterion, optimizer, trainloader, validloader, power)
FUtilities.save_checkpoint(model, optimizer, path, hidden_layers, learning_rate, epochs)

print ("done.")