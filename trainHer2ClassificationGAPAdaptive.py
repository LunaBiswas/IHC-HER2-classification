#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

#img_dir = '/nvme1_drive/IHC_Her2_data'
img_dir = '/media/chetan/Project/Datasets/IHCSegmentationTest/train'

model_path = './Her2Models/HNEHer2_net_GAP.pth'
          
batch_size = 16
num_epochs = 1

classes = ('HER2Positive', 'HER2Negative', 'HER2Intermediate', 'Others')

kernel_size = 9

class Net(nn.Module):
    # CNN model for HNE Her2 classification
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size, dilation = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, dilation = 2)
        self.gap = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(512, 4)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.gap(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x

def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([
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

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)

    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    trainloader, valloader = load_split_train_test(img_dir, .2)
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
    loss_same_count = 0
    loss_prev = 0
    val_accuracy_prev = 0

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print('Epoch ', epoch,':')
        running_loss = 0.0
        epoch_labels=[]
        epoch_predicted=[]
            
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
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
 
        if train_loss == loss_prev:
            loss_same_count += 1
        else:
            loss_same_count = 0
    
        if loss_same_count >= 10:
            break

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            net.eval()
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
            net.train()        
        val_accuracy = 100 * correct // total

        if val_accuracy > val_accuracy_prev:
            torch.save(net.state_dict(), model_path)
            print("Saved model ",model_path," for accuracy ",val_accuracy)

        print('Accuracy of the network on the val images: ', val_accuracy, ' %')
        print(confusion_matrix(epoch_labels,epoch_predicted))
        loss_prev = train_loss
        
    print('----------------------')
    print('Finished Training')
    print('----------------------')