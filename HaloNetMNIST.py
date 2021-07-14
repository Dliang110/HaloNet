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

class MNISTDataset(data.Dataset):
    def __init__(self, root="/home/mathiane/halonet-pytorch/MNISTImagesList.txt"):
        self.root = root
        with open(root, 'r') as f:
            content =  f.readlines()
        self.files_list = []
        for x in content:
            x =  x.strip()
            if x.find('reject') == -1:
                self.files_list.append(x)

        ## Image Transformation ##
        # High color augmntation
        # Random orientation
        self.transform = transforms.Compose([
            transforms.Resize((390,390)),
            transforms.CenterCrop(384),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            transforms.ToTensor(),
#            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def __getitem__(self,index):
        img =  Image.open(self.files_list[index])
        w, h = img.size
        ima = Image.new('RGB', (w,h))
        data = zip(img.getdata(), img.getdata(), img.getdata())
        ima.putdata(list(data))
        if self.transform is not None:
            img_c = self.transform(ima)
        label = self.files_list[index].split('/')[-2]
        label =   np.array(int(label))
        label = torch.from_numpy(label)
        return img_c, label

    def __len__(self):
        return len(self.files_list)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()
    train_dset = MNISTDataset()
    train_loader = torch.utils.data.DataLoader(
            train_dset,
            batch_size=2, shuffle=True,
            num_workers=4, pin_memory=False)
    model = Halonet.halonetB4()
    model.load_state_dict(torch.load("/home/mathiane/halonet-pytorch/saved_model_MNISTH4_1/HalonetH4_MNIST.pt"))
    model.cuda()
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    minloss = 1e10

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        t_loss = []
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            t_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                writer.add_scalar('Loss', loss.item(), len(train_loader) * epoch + (i + 1))
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        if np.mean(t_loss) <= minloss:
            minloss = np.mean(t_loss)
            os.makedirs('/home/mathiane/halonet-pytorch/saved_model_MNISTH4_1', exist_ok=True)
            torch.save(model.state_dict(), '/home/mathiane/halonet-pytorch/saved_model_MNISTH4_1/HalonetH4_MNIST.pt')

    print('Finished Training')
