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

# switch to evaluate mode
model.eval()

# if len(sys.argv) > 1:
#     image_path = sys.argv[1]
# else:
#     print("Please specify an input image")
try:
    image_path = sys.argv[1]
    image_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)
except:
    sys.exit("Invalid input image")

# image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
image_cv = cv2.resize(image_cv, (32, 32))
# cv2.imwrite('cv_img.jpg', image_cv)     # output image processed by cv for debugging
# print(image_cv.shape)
transform = transforms.ToTensor()
image_tensor = transform(image_cv)
image_tensor = image_tensor.unsqueeze(0)
# print(image_tensor.shape)

# src = mpimg.imread(image_path)[:,:,:3]
# newImg = np.array(Image.fromarray(np.uint8(src)).resize(size=(32,32)))
# imagebatch = newImg.reshape(-1,3,32,32)
# image_tensor = torch.from_numpy(imagebatch).float()
# print(image_tensor.shape)

# img1 = image_tensor[0]
# # img1 = img1.numpy()
# save_image(img1, 'img1.jpg')     # output image processed by tensor for debugging

torch.cuda.set_device(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("CPU or CUDA? " + device)
device = torch.device(device)

output = model(image_tensor.to(device))
_, predicted = torch.max(output.data, 1)
pre = predicted.cpu().numpy()
# print(pre)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(classes[pre[0]])
