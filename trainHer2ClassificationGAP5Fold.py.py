#!/usr/bin/env python
# coding: utf-8

from unittest import loader
import torch
import torchvision
import numpy as np
import torch.nn as nn
import tensorflow as tf
from cgitb import reset
import torch.optim as optim
from collections import Counter
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

img_dir = '/nvme1_drive/IHC_Her2_data'

batch_size = 32
num_epochs = 200

classes = ('HER2Positive', 'HER2Negative', 'HER2Intermediate', 'Others')

img_height = 238
img_width = 273
kernel_size = 9
conv_output_height = (((img_height - kernel_size + 1)//2) - kernel_size + 1)//2
conv_output_width = (((img_width - kernel_size + 1)//2) - kernel_size + 1)//2

class Net(nn.Module):
    # CNN model for HNE Her2 classification
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size, dilation = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, dilation = 2)
        self.gap = nn.AvgPool2d((conv_output_height, conv_output_width))
        self.fc1 = nn.Linear(32, 4)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.gap(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x

def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ])
    val_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      ])
    train_data = datasets.ImageFolder(datadir,       
                    transform=train_transforms)
    val_data = datasets.ImageFolder(datadir,
                    transform=val_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=batch_size)
    valloader = torch.utils.data.DataLoader(val_data,
                   sampler=val_sampler, batch_size=batch_size)
    return trainloader, valloader

def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'

    print(device)

    net = Net()
    net.to(device)
#    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    fold_accuracies=[0,0,0,0,0]
    loaders = load_split_train_test(img_dir, .2)
    
    for fold in range(5):
        
        trainloader, valloader = load_split_train_test(img_dir, .2)
        print('--------- Fold no: ', fold)
        labellist=[]
        
        for train_data in list(trainloader):
            labellist.extend(train_data[1].cpu().tolist())
        print('Class counts in train dataset:',dict(zip(list(labellist),[list(labellist).count(i) for i in list(labellist)])))
       
        labellist=[]
        for val_data in list(valloader):
            labellist.extend(val_data[1].cpu().tolist())
        print('Class counts in val dataset:',dict(zip(list(labellist),[list(labellist).count(i) for i in list(labellist)])))
       
        correct = 0
        total = 0

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            print('Epoch ', epoch,':')
            running_loss = 0.0
            epoch_labels=[]
            epoch_predicted=[]
            
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                #inputs, labels = data
                inputs, labels = data[0].to(device), data[1].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
        
            train_loss = running_loss/len(trainloader)
            print('Loss: ',train_loss)
 
            if epoch % 9 == 0:
                PATH = './Her2Models/HNEHer2_net_Fold_'+str(fold)+'_'+str(epoch)+'.pth'
                torch.save(net.state_dict(), PATH)

            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in valloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    # calculate outputs by running images through the network
                    outputs = net(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    epoch_labels +=  list(labels.cpu().numpy())
                    epoch_predicted += list(predicted.cpu().numpy())
                    
            val_accuracy = 100 * correct // total
            
            print('Accuracy of the network on the val images: ', val_accuracy, ' %')
            print(confusion_matrix(epoch_labels,epoch_predicted))

            if val_accuracy > fold_accuracies[fold]:
                fold_accuracies[fold] = val_accuracy

        net.apply(reset_weights)

    print('----------------------')
    print('Finished Training')
    print('----------------------')
    print('Accuracies in 5 folds: ',fold_accuracies)
    print('Average accuracy: ',np.mean(fold_accuracies))
    
    PATH = './Her2Models/HNEHer2_net_'+str(epoch)+'.pth'
    torch.save(net.state_dict(), PATH)
