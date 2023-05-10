import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
#import dataloader
import model_zeroDCEpp
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import cv2


def lowlight(data_lowlight):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    scale_factor = 1
    # data_lowlight = Image.open(image_path)
    #
    data_lowlight = (np.asarray(data_lowlight) / 255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()

    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    data_lowlight = data_lowlight[0:h, 0:w, :]
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net = model_zeroDCEpp.enhance_net_nopool(scale_factor).cuda()
    DCE_net.load_state_dict(torch.load(r"C:\Users\rom21\OneDrive\Desktop\git_project\Final_project_afeka\afeka_project_image_classification_low_light\code\ZeroDCEpp\snapshots_Zero_DCE++\Epoch99.pth"))
    start = time.time()
    enhanced_image, params_maps = DCE_net(data_lowlight)

    end_time = (time.time() - start)

    #print(end_time)
    #image_path = image_path.replace('test_data', 'result_Zero_DCE++')

    # result_path = image_path
    # if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
    #     os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
    # import pdb;pdb.set_trace()
    #torchvision.utils.save_image(enhanced_image, result_path)
    enhanced_image_out = np.zeros(img.shape)
    x = np.squeeze(enhanced_image.cpu().detach().numpy())
    enhanced_image_out[:, :, 0] = x[0]
    enhanced_image_out[:, :, 1] = x[1]
    enhanced_image_out[:, :, 2] = x[2]
    enhanced_image_out = np.round(enhanced_image_out * 255)
    return enhanced_image_out


if __name__ == '__main__':

    with torch.no_grad():
        images_name = r"person\000000000872.jpg"
        image = rf"E:\coco_aug_1\{images_name}"
        img = cv2.cvtColor(cv2.imread(rf"E:\coco_aug_1\{images_name}"), cv2.COLOR_BGR2RGB) * 1.

        e = lowlight(img)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(img.astype(np.uint8))
        plt.figure()
        plt.imshow(e.astype(np.uint8))
        # filePath = 'data/test_data/'
        # file_list = os.listdir(filePath)
        # sum_time = 0
        # for file_name in file_list:
        #     test_list = glob.glob(filePath + file_name + "/*")
        # for image in test_list:
        #     print(image)
        #     sum_time = sum_time + lowlight(image)
        #
        # print(sum_time)

