import os
import json
from xml.dom import InvalidAccessErr
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import argparse

from mv3d import config
from mv3d.eval import config as eval_config

#imports from demo.py
from torchvision import transforms

import PIL
from PIL import Image
import matplotlib.pyplot as plt

import os.path
from pathlib import Path
import glob
import sys

import pdb

from modules.unet import UNet
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform

def save_outputs(img_path, output_file_name):
    with torch.no_grad():
        save_path = os.path.join(args.output_path, f'{output_file_name}_depth.png')

        print(f'Reading input {img_path} ...')
        img = Image.open(img_path)

        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

        rgb_path = os.path.join(args.output_path, f'{output_file_name}_rgb.png')
        trans_rgb(img).save(rgb_path)

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3,1)

        output = model(img_tensor).clamp(min=0, max=1)

        class_result = classify(output.cpu().numpy())

        output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
        output = output.clamp(0,1)
        output = 1 - output
        plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')
        
        print(f'Writing output {save_path} ...')
        print(f'classification result: {class_result}')


def classify(depth_preds):
    threshold = 0.02

    ranges = np.arange(0, 1.02, 0.02) #omnidata depths are normalized from 0 to 1, window size 0.02
    n_ranges = ranges.shape[0] - 1

    H = W = 384
    n_total_pixels = H*W

    f = open("results.txt",'w')

    mean = 0
    all_prob = []

    for i in range(n_ranges):
        
        min_depth = ranges[i]
        max_depth = ranges[i+1]

        valid_pixels = ((depth_preds > min_depth) & (depth_preds <= max_depth)).astype(np.float32)
        n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))
        perc_valid_pixels = (n_valid_pixels / n_total_pixels) * 100

        avg_perc_valid = np.mean(perc_valid_pixels)
        all_prob.append(round(avg_perc_valid,3)) 
        print(f'Range {min_depth:.2f} - {max_depth:.2f} | perc valid: {avg_perc_valid:6.3f} %')

        res = [f'{avg_perc_valid:6.3f}', '\n']
        f.writelines(res)

        #calculate mean
        mean += max_depth*avg_perc_valid

    # calculate percentage of invalid pixels------------------------------------------
    invalid_pixels = (depth_preds == 0).astype(np.float32)
    n_invalid_pixels = np.sum(invalid_pixels, axis=(1,2))
    perc_invalid_pixels = (n_invalid_pixels / n_total_pixels) * 100
    avg_perc_invalid = np.mean(perc_invalid_pixels)
    print(f'perc_invalid: {avg_perc_invalid:6.3f} %')

    #calculate mean & std-------------------------------------------------------------
    mean /= 100 #divide by 100 to get %
    std = 0
    x = np.arange(0.02, 1.02, 0.02)

    for i, prob in zip(x, all_prob):
        std += pow((i-mean),2)*(prob/100)
    std = pow(std, 0.5)
    print(f'mean: {mean:.3f}')
    print(f'standard deviation of the probability distribution: {std:.3f}')

    #classifies
    if (std > threshold):
        return 1
    else:
        return 0

#used to parse a file into an array, data separated by space
# def parse_file(filename):
#     with open(filename, "r") as f:
#         data = [[float(y) for y in x.strip().split(" ")] for x in f]
#         data[0] = [int(x) for x in data[0]]

#         return data

#---------------------------------------------------------------------------------
# MAIN FUNCTION 
#---------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

# parser.add_argument('--task', dest='task', help="normal or depth")
# parser.set_defaults(task='NONE')

parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
parser.set_defaults(im_name='NONE')

parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
parser.set_defaults(store_name='NONE')

#get testing data ---------
# parser.add_argument('--testing_file', dest='testing_file', help="path to file containing testing data")
# parser.set_defaults(test_name='NONE')

# data = parse_file(args.testing_file)
#--------------------------

args = parser.parse_args()

root_dir = './pretrained_models/'

trans_topil = transforms.ToPILImage()

os.system(f"mkdir -p {args.output_path}")
map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_size = 384
pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
# model = DPTDepthModel(backbone='vitl16_384') # DPT Large
model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
if 'state_dict' in checkpoint:
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k[6:]] = v
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)
model.to(device)
trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=0.5, std=0.5)])

trans_rgb = transforms.Compose([transforms.Resize(512, interpolation=PIL.Image.BILINEAR),
                                transforms.CenterCrop(512)])

#save results
img_path = Path(args.img_path)
if img_path.is_file():
    save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
elif img_path.is_dir():
    for f in glob.glob(args.img_path+'/*'):
        save_outputs(f, os.path.splitext(os.path.basename(f))[0])
else:
    print("invalid file path!")
    sys.exit()




