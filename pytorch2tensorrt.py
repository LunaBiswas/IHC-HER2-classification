import torch
import numpy as np
import torch.nn as nn
import torch_tensorrt
import torch.nn.functional as F

### Hyper-Parameters
tile_height = 238 # height of each tile 
tile_width = 273 # width of each tile

### Model
kernel_size = 9

torch_model_path = '/nvme1_drive/Her2/code/Her2Models/HNEHer2_net_GAP.pth'
trt_model_path = '/nvme1_drive/Her2/code/Her2Models/HNEHer2_net_GAP_trt_ts.ts'

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

model = Net()
model.load_state_dict(torch.load(torch_model_path))
model = Net().eval()  # torch module needs to be in eval (not training) mode

inputs = [
    torch_tensorrt.Input(
        min_shape=[35, 3, 238, 273],
        opt_shape=[35, 3, 238, 273],
        max_shape=[35, 3, 256, 273]
    )
]
enabled_precisions = {torch.float}  # Run with fp16

trt_ts_module = torch_tensorrt.compile(
    model, inputs=inputs, enabled_precisions=enabled_precisions
)

torch.jit.save(trt_ts_module, trt_model_path)