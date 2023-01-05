import os
import gc
import sys
import time
import json
import torch
import cupy as cp
import numpy as np
import pandas as pd
import pickle as pkl
import torch.nn as nn
from tqdm import tqdm
from glob import glob
import torch_tensorrt
from cucim import CuImage
import torch.nn.functional as F
from torchvision import transforms
from cucim.skimage.color import rgb2hsv

### Hyper-Parameters
level = 0 #40x
patch_size = '238x273'
tissue_threshold = 30
num_workers = 4
write_file_flag = True
write_json_flag = True
pramana_json_flag = True
tile_classes = ['Intermediate', 'Negative', 'Other', 'Positive']

num_rows = 5 # number of rows in the aoi
num_cols = 7 # number of cols in the aoi
num_tiles = num_rows * num_cols # number of tiles in an aoi 

tile_height = 238 # height of each tile 
tile_width = 273 # width of each tile

### Device
device = 0
device0 = torch.device("cuda:0")

### Model
kernel_size = 9

class Net(nn.Module):
    # CNN model for HNE Her2 classification
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size, dilation = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, dilation = 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(32, 4)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.gap(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x

### Function definitions

def row_major_torch128_patchify(region):
    '''
    Creates 35 patches of size tile_height x tile_width from an aoi
    '''

    # region = torch.tensor(cp.transpose(region,(2,0,1)))
    region = cp.transpose(region,(2,0,1))
    patches = cp.zeros((1,num_tiles,3,tile_height,tile_width),dtype = cp.float32)
 
    patch_num = 0
    for w in range(num_cols):
        for h in range(num_rows):
            patches[:,patch_num,:,:,:] = region[:,h*tile_height:(h+1)*tile_height,w*tile_width:(w+1)*tile_width]
            patch_num += 1
 
    return patches

def pytorch_only_model_forward(arr,model,device):
    '''
    returns class label predictions and timelog from the classification model
    '''
    #scaling and move to GPU
    arr = (arr/255.0).to(device)
    # normalize
    arr = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(arr)
    # unified model forward pass
    # print("MODEL INPUT SHAPE  ",arr.shape)
    x_tile_class = model(arr)
    # print(x_tile_class.shape)
    torch.cuda.synchronize()
    # argmax for deciding classes
    class_labels = torch.argmax(x_tile_class,1)
   # print(class_labels)
    return class_labels

### Main function
if __name__=='__main__':
    
    tissue_type = "HER2"
    csv_file_path = "/media/adminspin/558649a1-21fd-43b8-b53d-7334f802b47a/wsi_tb_data/HER2DATA/csv/24dec_26dec_formatted_her2_only.csv"

    #tissue_type = sys.argv[1].upper()    # Accept tissue type and csv file (with WSI names) as command line arguments
    #csv_file_path = sys.argv[2]
   #
    df = pd.read_csv(csv_file_path)      # Filters in wsi names for the tissue type entered      
    slide_list = list(df['slide_name'][df['type']==tissue_type])
   
    #specify a wsi_path or wsi_directory having multiple paths
    base_tiff_path = "/media/adminspin/558649a1-21fd-43b8-b53d-7334f802b47a/wsi_tb_data/HER2DATA/tiff/"+tissue_type+"/"
    
    #trt_model_path = '/nvme1_drive/Her2/code/Her2Models/HNEHer2_net_GAP_trt_ts.ts'

    wsi_list = [base_tiff_path+x for x in slide_list]

    #her2_model = torch.jit.load(trt_model_path)
    her2_model = Net()
    model_path = '/media/adminspin/558649a1-21fd-43b8-b53d-7334f802b47a/wsi_tb_data/Her2/code/Her2Models/HNEHer2_GAP.pth'
    her2_model.load_state_dict(torch.load(model_path))
    her2_model.eval().to(device)

    pipeline_file_path = '/media/adminspin/558649a1-21fd-43b8-b53d-7334f802b47a/wsi_tb_data/Her2/code/tiff_processing_Her2_log.txt'
    pipeline_data = {"storage_init":[], "gpu_mapping":[], "model":[], "result_assign":[],\
                    "saving_tiles_feat":[],"aoi_process":[]}

 #   print('wsi list:',wsi_list)
    for wsi_source in wsi_list:
        print("Checking wsi source :", wsi_source)
        if os.path.isdir(wsi_source): 
            feature_store_path = wsi_source
            wsi_path_list = sorted(glob(wsi_source+"/*.tif"))
        else:
            feature_store_path = os.path.dirname(wsi_source)
            wsi_path_list = [wsi_source]
        print('feature store path',feature_store_path)
        
        for wsi_path in wsi_path_list:
            print("wsi path : ",wsi_path)
            NOW = time.time()
            wsi_name = str(wsi_path).split('/')[-1]
            if os.path.exists(feature_store_path + '/' + str(wsi_name.split('.tif')[0])+'/'+str(wsi_name.split('.tif')[0])+'_'+str(patch_size)+'cluser_id(row)_tile_class(columns)_distribution.pkl'):
                print("SKIPPING {}",wsi_name)
                continue
            else:
                wsi = CuImage(wsi_path)
                x_size,y_size = wsi.resolutions["level_dimensions"][level]
                # iterator on AOIs in a WSI : 1192, 1912
                start_loc_list = [(sx, sy) for sx in range(0, x_size, 1912) for sy in range(0, y_size, 1192)]
                it_region0 = wsi.read_region(start_loc_list,(1912,1192), level = level,batch_size=1, num_workers=num_workers,drop_last=True)
                #total number of AOIs
                batch_len = len(it_region0)
                # Memory allocation for holding tile_class
                wsi_tileclass_array = -1*np.ones((batch_len*num_tiles,2),dtype = np.float32)
                
            with torch.no_grad():
                with cp.cuda.Device(0):
                    for i,aoi in tqdm(enumerate(it_region0)):
                        # initialize memory
                        t1 = time.time()
                        aoi_tileclass = -1*torch.ones((num_tiles,1),dtype=torch.float32,device = device0)
                        patches = torch.zeros((num_tiles,3,tile_height,tile_width),dtype = torch.float, device = device0)
                        t2 = time.time()
                        # copy to GPU
                        region_aoi = cp.asarray(aoi)

                        patches = row_major_torch128_patchify(region_aoi)
                        patches = torch.as_tensor(patches, device='cuda')
                        patches = torch.squeeze(patches,0)
                        t3 = time.time()
                        # predicting class_labels of each tile                        
                        aoi_tileclass[:,0] = pytorch_only_model_forward(patches,her2_model,device0)
                        t4 = time.time()
                        
                        #appending aoi feature in wsi features (tile_class)
                        t5 = time.time()
                        wsi_tileclass_array[i*num_tiles:(i+1)*num_tiles,:] = aoi_tileclass.cpu()
                        # torch.cuda.synchronize()                    
                        t6 = time.time()

                        pipeline_data['storage_init'].append(t2-t1)
                        pipeline_data['gpu_mapping'].append(t3-t2)
                        pipeline_data['model'].append(t4-t3)
                        pipeline_data['result_assign'].append(t6-t5)
                        pipeline_data['aoi_process'].append(t6-t1)
                   
                new = time.time()
     
                ## WRITING THE file
                os.makedirs(feature_store_path + '/'+str(wsi_name.split('.tif')[0]),exist_ok = True)
                if write_file_flag:        
                    print('WRITING')
                    t7 = time.time()
                    wsi_tile_cluster_path = feature_store_path + '/' + str(wsi_name.split('.tif')[0])+'/'+str(wsi_name.split('.tif')[0])+'_'+str(patch_size)+'cluser_id(row)_tile_class(columns)_distribution.pkl'
                    with open(wsi_tile_cluster_path,'wb') as f:
                        pkl.dump(aoi_tileclass,f)
                    pipeline_data['saving_tiles_feat'].append(time.time()-t7)
                
            #    print("Processed wsi :", wsi_path)

            # get the arrays having all the labels feature_array has: all tile classifcation labels
            tile_class_label = wsi_tileclass_array[:,0]
            del wsi_tileclass_array
            gc.collect()

            color = [-3342337,-1657882,-256,-16776961]
            
            #JSON WRITING CODE
            if write_json_flag:
                print(wsi_name)

                # Using the wsi, x_size, y_size, and start_loc_list from the above
                print('***********************') 
                
                lines = "{\n"
                lines += '"type": "FeatureCollection",\n'
                lines += '"features": [\n'

                for k,(x, y) in enumerate(start_loc_list):
                    for ix in range(7):
                        x_ij = x + tile_width*ix
                        
                        if x_ij+tile_width > x_size:
                            x_n = x_size
                        else:
                            x_n = x_ij+tile_width

                        for jy in range(5):
                            y_ij = y + tile_height*jy
                            if y_ij +tile_height > y_size:
                                y_n = y_size
                            else:
                                y_n = y_ij +tile_height
                            
                            b1 = (x_ij,y_ij)
                            b2 = (x_ij,y_n)
                            b3 = (x_n,y_n)
                            b4 = (x_n,y_ij)
                            b5 = (x_ij,y_ij)
                            lines += "{\n"
                            lines += '"type": "Feature",\n'
                            lines += '"geometry": {\n'
                            lines += '"type": "Polygon",\n'
                            lines += '"coordinates": [\n'
                            lines += '[\n'
                            lines += f'[{b1[0]}, {b1[1]}],\n'
                            lines += f'[{b2[0]}, {b2[1]}],\n'
                            lines += f'[{b3[0]}, {b3[1]}],\n'
                            lines += f'[{b4[0]}, {b4[1]}],\n'
                            lines += f'[{b5[0]}, {b5[1]}]\n'
                            lines += ']\n'
                            lines += ']\n'
                            lines += '},\n'
                            lines += '"properties": {\n'
                            lines += '"object_type": "annotation",\n'
                            lines += '"classification": {\n'
                            
                            label = int(tile_class_label[k*35+ix*5+jy*1])
                            lines += f'"name": "{tile_classes[int(tile_class_label[k*35+ix*5+jy*1])]}",\n'
                            lines += f'"colorRGB": {color[int(tile_class_label[k*35+ix*5+jy*1])]}\n'

                            lines += '},\n'
                            lines += '"isLocked": false\n'
                            lines += '}\n'
                            
                            
                            if k*35+ix*5+jy*1 == len(start_loc_list)*7*5 -1:
                                lines += '}\n' 
                            else: lines += '},\n'
                                                            
                lines += ']\n'
                lines += '}'
                method = '4class_tile_'
  #              print('Saving jsons',feature_store_path+ '/' + str(wsi_name.split('.tif')[0])+'/'+str(method)+str(wsi_name)+'.geojson')
                with open(feature_store_path+ '/' + str(wsi_name.split('.tif')[0])+'/'+str(method)+str(wsi_name)+'.geojson', 'w') as f:
                    f.write(lines)
                del lines
            
            # PRAMANA TUMOR JSON DATA
            if pramana_json_flag:
                print('***********************') 
                pramana_tumor_json_data = {"status": "success", "label_name": [], \
                 "size": {"x": "180", "y": "180"}, "data":[]}

                for n in range(len(color)):
                    pramana_tumor_json_data["label_name"].append({"name":tile_classes[n], "color":color[n], "value":n})
              
                for k,(x, y) in enumerate(start_loc_list):
                    for ix in range(7):
                        x_ij = x + tile_width*ix
                        
                        if x_ij+tile_width > x_size:
                            x_n = x_size
                        else:
                            x_n = x_ij+tile_width

                        for jy in range(5):
                            y_ij = y + tile_height*jy
                            if y_ij +tile_height > y_size:
                                y_n = y_size
                            else:
                                y_n = y_ij +tile_height
                            
                            if tile_class_label[k*35+ix*5+jy*1] != -1:
                                pramana_tumor_json_data["data"].append([int((x_ij+x_n)/2), int((y_ij+y_n)/2), int(tile_class_label[k*35+ix*5+jy*1])])

                method = '4class_tile_'
            
     #           print('Saving jsons',feature_store_path+ '/' + str(wsi_name.split('.tif')[0])+'/Pramana_'+method+str(wsi_name)+'.json')
                pramana_tumor_json_object = json.dumps(pramana_tumor_json_data, indent=4)
                with open(feature_store_path+ '/' + str(wsi_name.split('.tif')[0])+'/Pramana_'+method+str(wsi_name)+'.json', 'w') as outfile:
                    outfile.write(pramana_tumor_json_object)

            del pramana_tumor_json_data
            del pramana_tumor_json_object
            
            NEW = time.time()
            print("***********************")
            print("Total time:",NEW-NOW)
            print("***********************")
            with open(pipeline_file_path,'w') as f:
                f.write(json.dumps(pipeline_data))

            print("average aoi process time:",np.mean(pipeline_data['aoi_process']))
            