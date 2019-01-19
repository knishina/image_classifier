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
parser.add_argument("image_path", action="store", nargs="*", help="Default is flowers/test/80/image_01983.jpg", default="flowers/test/80/image_01983.jpg")
parser.add_argument("--gpu", dest="gpu", action="store", help="Default is gpu", default="gpu")
parser.add_argument("--checkpoint_path", action="store", dest="checkpoint_path", help="Default is densenet121_checkpoint.pth", default="./densenet121_checkpoint.pth")
parser.add_argument("--topk", action="store", dest="topk", type=int, help="Default is 5", default=5)
parser.add_argument("--cat_to_name", action="store", dest="cat_to_name", help="Default is cat_to_name", default="cat_to_name")


par = parser.parse_args()
image_path = par.image_path
topk = par.topk
power = par.gpu
checkpoint_path = par.checkpoint_path
cat_to_name = par.cat_to_name

trainloader, testloader, validloader = FUtilities.load_data()
FUtilities.load_checkpoint(checkpoint_path)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

FUtilities.predict(image_path, model, topk)
FUtilities.output(image_path, model)

print ("done.")