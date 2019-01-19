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


parser = argpars.ArgumentParser()
parser.add_argument("data_dir", nargs = "*", action="store", help="Store data in folder with subfolders labeled test, train, valid.  Input name of the primary folder here and the program will extract the rest.", default= "./flowers/")
parser.add_argument("--gpu", dest="gpu", action="store", help="Default is gpu", default="gpu")
parser.add_argument("--checkpoint_path", dest="checkpoint_path", action="store", help="Default is densenet121_checkpoint.pth", default="./densenet121_checkpoint.pth")
parser.add_argument("--hidden_layers", dest="hidden_layers", action="store", type=int, help="Default is 512", default=512)
parser.add_argument("--learning_rate", dest= "learning_rate", action="store", type=int, help="Default is 0.001", default=0.001)
parser.add_argument("--epochs", dest="epochs", action="store", type=int, help="Default is 10", default=10)

par = parser.parse_args()
where = par.data_dir
checkpoint_path = par.save_dir
learning_rate = par.learning_rate
hidden_layers = par.hidden_layers
power = par.gpu
epochs = par.epochs



trainloader, validloader, testloader = FUtilities.load_data(where)
model, optimizer, criterion = FUtilities.build_model(hidden_layers, learning_rate, class_to_idx, power)
FUtilities.train_model(model, learning_rate, criterion, optimizer, trainloader, validloader, power)
FUtilities.save_checkpoint(checkpoint_path)

print ("done.")