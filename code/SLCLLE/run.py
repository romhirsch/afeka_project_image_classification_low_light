import sys
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import lowlight_model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.colors
from scipy.signal import convolve, convolve2d, correlate2d
import scipy, scipy.misc, scipy.signal
import matplotlib.pyplot as plt

class scl_lle:
  def __init__(self):
    self.SCL_LLE_net = lowlight_model.enhance_net_nopool().cuda()
    self.SCL_LLE_net.load_state_dict(torch.load('checkpoints/SCL-LLE.pth'))

  def test(self, img):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    data_lowlight = img
    data_lowlight = (np.asarray(data_lowlight)/255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    _, enhanced_image, _ = self.SCL_LLE_net(data_lowlight)

    return enhanced_image

# mynet = scl_lle()
# img = cv2.cvtColor(cv2.imread(r"E:\coco_aug_4\bus\004304.jpg"), cv2.COLOR_BGR2RGB)
# x = mynet.test(img)
# x = x.cpu().detach().numpy()
#
# img2 = np.squeeze(x)
# img_scl = np.zeros(img.shape)
# img_scl[:, :, 0] = img2[0, :, :]
# img_scl[:, :, 1]  = img2[1, :, :]
# img_scl[:, :, 2]  = img2[2, :, :]




