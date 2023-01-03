#!/usr/bin/env python
# coding: utf-8

from cgitb import reset
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler


img_dir = '/nvme1_drive/IHC_Her2_data'

batch_size = 16
num_epochs = 100

classes = ('HER2Positive', 'HER2Negative', 'HER2Intermediate', 'Others')

img_height = 238
img_width = 273
kernel_size = 9
conv_output_height = (((img_height - kernel_size + 1)//2) - kernel_size + 1)//2
conv_output_width = (((img_width - kernel_size + 1)//2) - kernel_size + 1)//2
 
linear_input_size = conv_output_height * conv_output_width

class Net(nn.Module):
    # CNN model for HNE Her2 classification
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size)
        self.fc1 = nn.Linear(16 * linear_input_size, 2048)
        self.fc2 = nn.Linear(2048, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

if __name__ == '__main__':

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'

    print(device)

    net = Net()
    net.to(device)
#    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
 
    k_fold = KFold(n_splits=5)

    train_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ])
    train_data = datasets.ImageFolder(img_dir,       
                    transform=train_transforms)

    fold_accuracies=[0,0,0,0,0]

    for fold, (train_idx,val_idx) in enumerate(k_fold.split(train_data)):
        print('--------- Fold no: ', fold)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=batch_size)
        valloader = torch.utils.data.DataLoader(train_data,
                   sampler=val_sampler, batch_size=batch_size)
 
        correct = 0
        total = 0

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            print('Epoch ', epoch,':')
            running_loss = 0.0
            TP = np.zeros(len(classes))
            TN = np.zeros(len(classes))
            FP = np.zeros(len(classes))
            FN = np.zeros(len(classes))

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
            writer.add_scalar('Loss/train', train_loss, epoch)

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
                    for i in range(len(classes)):
                        TP[i] += ((predicted == labels)*(predicted == i)).sum().item()
                        TN[i] += ((predicted == labels)*(predicted != i)).sum().item()
                        FP[i] += ((predicted != labels)*(predicted == i)).sum().item()
                        FN[i] += ((predicted != labels)*(predicted != i)).sum().item()

            val_accuracy = 100 * correct // total
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            writer.add_scalar('TP/val', np.mean(TP), epoch)
            writer.add_scalar('TN/val', np.mean(TN), epoch)
            writer.add_scalar('FP/val', np.mean(FP), epoch)
            writer.add_scalar('FN/val', np.mean(FN), epoch)

            print('Accuracy of the network on the val images: ', val_accuracy, ' %')
            print('True Positive:',TP)
            print('True Negative:',TN)
            print('False Positive:',FP)
            print('False Negative:',FN)

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
