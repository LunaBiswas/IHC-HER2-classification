#!/usr/bin/env python
# coding: utf-8

import cv2
import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F

import os
import glob
from PIL import Image
import numpy as np

#Configurations
testImagePath='/media/adminspin/558649a1-21fd-43b8-b53d-7334f802b47a/wsi_tb_data/Her2/IHC_Her2_data_infer/'
outPath='/media/adminspin/558649a1-21fd-43b8-b53d-7334f802b47a/wsi_tb_data/Her2/code/output/PIL/'

batchSize=1

DEVICE = torch.device('cuda:0') # device used for training and evaluation
#DEVICE = torch.device('cpu')

num_workers = 2
pin_memory = True

classes = ('Intermediate','Negative','Other','Positive')

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

def save_prediction_as_image(
    image, filename, model, folder=outPath, device=DEVICE
):
    model.eval()
    
    image = image.to(device=device).float()
    with torch.no_grad():
        # calculate outputs by running images through the network
        outputs = model(image.unsqueeze(0))
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
            
        if not os.path.exists(outPath+classes[predicted.item()]):
            os.mkdir(outPath+classes[predicted.item()])
        #cv2.imwrite(outPath+str(predicted.item())+"/"+images[idx], cv2.imread(testImagePath+images[idx]))
        print('Trying to save image at ', outPath+classes[predicted.item()]+"/"+filename)
        save_image(image,outPath+classes[predicted.item()]+"/"+filename)
        # save_image(
        #     cv2.imread(testImagePath+images[idx]),f"{folder}/{classes[predicted[idx]]}/{images[idx]}"
        # )
    
    model.train()

if __name__ == '__main__':
    
    print('Started prediction.')
   
    # load model
    net = Net().to(DEVICE)
    net.load_state_dict(torch.load("/media/adminspin/558649a1-21fd-43b8-b53d-7334f802b47a/wsi_tb_data/Her2/code/Her2Models/HNEHer2_net_GAP_495.pth"))
    net.eval()
    
    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ])
    
    for file in glob.glob(testImagePath+'*.ppm'):
        image = Image.open(file)
        image = np.asarray(image).astype('float32')  # Convert from integers to floats

        # Normalize to the range 0-1
        image /= 255.0
        image = test_transform(image)
        filename = file.split('/')[-1]
        print('File = ', file, 'Filename =', filename)

        # save segmented outputs
        save_prediction_as_image(
            image, filename, net, folder=outPath, device=DEVICE
        )

    net.train()
    print("Prediction complete!")
