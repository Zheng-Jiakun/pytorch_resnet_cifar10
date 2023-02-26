import sys

import torch
import torch.nn as nn
import resnet

# import matplotlib.image as mpimg  # mpimg for reading images
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
import cv2
import torchvision.transforms as transforms
from torchvision.utils import save_image

checkpoint = torch.load("save_resnet20/checkpoint.th")
# checkpoint = torch.load("save_resnet56/checkpoint.th")
# checkpoint = torch.load("pretrained_models/resnet1202-f3b1deed.th")

# print(type(checkpoint))
# print(checkpoint)

model = torch.nn.DataParallel(resnet.__dict__["resnet20"]())
# model = torch.nn.DataParallel(resnet.__dict__["resnet56"]())
# model = torch.nn.DataParallel(resnet.__dict__["resnet1202"]())
model.cuda()

model.load_state_dict(checkpoint['state_dict'])

# for layers in model.children():
# layers = next(model.children())
# print(layers.conv1)

# print(help(model))
# print(model.state_dict().keys())

# print(model.state_dict())

for layers in model.children():
    for key in layers.state_dict():
        print('key: ', key)
        param = layers.state_dict()[key]
        
        param_np = param.cpu().numpy()
        print('share: ', param_np.shape)
        if param_np.shape != ():
            for num in param_np:
                print(num)
        else:
            print(num)
        print('=====')
    print('========end========')
