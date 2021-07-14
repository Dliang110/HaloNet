import Halonet
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from collections import OrderedDict
from itertools import chain
import torch.utils.data as data
import random
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from HaloNetMNIST import MNISTDataset

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()
    test_dset = MNISTDataset( root="/home/mathiane/halonet-pytorch/MNIST_filepath_test.txt")
    test_loader = torch.utils.data.DataLoader(
            test_dset,
            batch_size=2, shuffle=True,
            num_workers=4, pin_memory=False)
    model = Halonet.halonetB4()
    model.load_state_dict(torch.load("/home/mathiane/halonet-pytorch/saved_model_MNISTH4_1/HalonetH4_MNIST.pt"))
    model.cuda()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():

        for i, data in enumerate(test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
print('Accuracy of the network on the %d test images: %d %%' % ( len(test_loader),
    100 * correct / total))
