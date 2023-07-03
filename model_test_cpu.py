import sys
import math
import numpy as np

import torch
import torch.nn as nn
import resnet

import cv2
import torchvision.transforms as transforms
from torchvision.utils import save_image

checkpoint = torch.load("save_resnet20/checkpoint.th")

model = torch.nn.DataParallel(resnet.__dict__["resnet20"]())
model.cuda()

model.load_state_dict(checkpoint['state_dict'])

class DynanmicQuantization:
    def __init__(self, valid_bits, threshold) -> None:
        self.valid_bits = valid_bits
        self.threshold = threshold
        pass
    
    def round(x, flag, delta, bits):
        if flag:
            min_val = - math.pow(2.0, bits - 1)
            max_val = math.pow(2.0, bits - 1) - 1
        else:
            min_val = 0
            max_val = math.pow(2.0, bits) - 1
        return torch.clamp(torch.floor(x / delta + 0.5), min=min_val, max=max_val) , delta

    def conv2d(input, kernels, stride, padding, bias):
        for kernel in kernels:
            for pos in np.arange(0, (input.shape(0) / stride - kernel.size(0) - 1), 1):
                output
        output = input
        return output

    def bn2d(input, weight, bias, mean, variance):
        output = input
        return output

    def relu(input):
        output = max(0, input)
        return output

    def linear(input):
        output = input
        return output

    def avg_pool2d(input):
        output = input
        return output

try:
    image_path = sys.argv[1]
    image_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)
except:
    sys.exit("Invalid input image")

image_cv = cv2.resize(image_cv, (32, 32))
print(type(image_cv))


# for layers in model.children():
#     for key in layers.state_dict():
#         print('key: ', key)
#         param = layers.state_dict()[key]
        
#         param_np = param.cpu().numpy()
#         print('share: ', param_np.shape)
#         if param_np.shape != ():
#             for num in param_np:
#                 print(num)
#         else:
#             print(num)
#         print('=====')
#     print('========end========')
