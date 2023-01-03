#!/usr/bin/env python
# coding: utf-8

from hashlib import new
import json
from cucim import CuImage
import os
import torch
import torchvision
from torchvision import models
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import pickle as pkl


json_list = ['/media/adminspin/558649a1-21fd-43b8-b53d-7334f802b47a/wsi_tb_data/HER2DATA/tiff/HER2/sideStains_JR-19-6192-B1-5_H01EBB55P-39395.ome.tif - Image0.geojson']

class_dictmap = {'other':'Other',
'Her2Positive':'Positive',
'Negative':'Negative',
'Her2Intermediate':'Intermediate',
}

original_tonumeric ={
'other1':0,
'Her2Positive':1,
'Her2Intermediate':2,
'Her2Negative':3,
}

feature_all = []
labels = []
from PIL import Image
for json_file in json_list:
    # print(json_file.split(' - Image0.geojson')[0])
    wsi_name = '/media/adminspin/558649a1-21fd-43b8-b53d-7334f802b47a/wsi_tb_data/HER2DATA/tiff/HER2/JR-19-6192-B1-5_H01EBB55P-39395/JR-19-6192-B1-5_H01EBB55P-39395.ome.tif'
    wsi = CuImage(wsi_name)
    json_raw = json.load(open(json_file))
    feature = np.zeros((1024),dtype = np.float32)
    for i in range(len(json_raw['features'])):
           
            location = json_raw['features'][i]['geometry']['coordinates'][0][0]
            patch = wsi.read_region((location[0],location[1]), [273, 238], 0)
   #         cocentric_patch = wsi.read_region((location[0]-128,location[1]-128), [512, 512], 0)
            #print(cocentric_patch.dtype,cocentric_patch.shape)
      #      class_name = class_dictmap[class_name]
            #class_name_new = original_tonumeric[class_name]
            
            os.makedirs('/media/adminspin/558649a1-21fd-43b8-b53d-7334f802b47a/wsi_tb_data/Her2/code/output/new/',exist_ok = True)
           # os.makedirs('/nvme1_drive/patch_cocentric_classification/20x/'+str(class_name)+'/',exist_ok = True)
            patch.save('/media/adminspin/558649a1-21fd-43b8-b53d-7334f802b47a/wsi_tb_data/Her2/code/output/new/'+ wsi_name.split('/')[-2] +'_'+str(location[0])+'_'+str(location[1])+'.ppm')
  #          print('/nvme1_drive/IHC_Her2_data/'+str(class_name)+'/'+str(wsi_name)+'_'+str(class_name)+'_'+str(location[0])+'_'+str(location[1])+'.ppm saved')
  #          cocentric_patch.save('/nvme1_drive/patch_cocentric_classification/20x/'+str(class_name)+'/'+str(wsi_name)+'_'+str(class_name)+'_'+str(location[0])+'_'+str(location[1])+'.ppm')

            #feature[:512],feature[512:1024] = trt_only_model_forward(patch,model_trt).cpu(),trt_only_model_forward(cocentric_patch,model_trt).cpu()
            #feature[-1] = int(class_name_new)
            #labels.append(class_name_new)
            #feature_all.append(feature)
            #print(class_name_new,class_name,feature_all[-1][-1])     
       #     print(class_name,location[0],location[1])      
            print("Saved to:",'/media/adminspin/558649a1-21fd-43b8-b53d-7334f802b47a/wsi_tb_data/Her2/code/output/new/'+ wsi_name.split('/')[-2] + '_' + str(location[0])+'_'+str(location[1])+'.ppm')      
                   
