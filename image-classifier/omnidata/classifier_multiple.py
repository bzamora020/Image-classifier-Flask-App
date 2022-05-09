import os
import shutil
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

output_path = "/data/ersp21/classifier/outputs"

#this function is used to save the depth distributions of failure cases into individual txt files
def save_depthDistributions(depth_preds, path):
    ranges = np.arange(0, 1.02, 0.02) #omnidata depths are normalized from 0 to 1, window size 0.02
    n_ranges = ranges.shape[0] - 1

    H = W = 384
    n_total_pixels = H*W

    f = open(path,'w')

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

        # print(f'{avg_perc_valid}\n')

        res = [f'{avg_perc_valid:6.3f}', '\n']
        f.writelines(res)
    
    # calculate percentage of invalid pixels------------------------------------------
    invalid_pixels = (depth_preds == 0).astype(np.float32)
    n_invalid_pixels = np.sum(invalid_pixels, axis=(1,2))
    perc_invalid_pixels = (n_invalid_pixels / n_total_pixels) * 100
    avg_perc_invalid = np.mean(perc_invalid_pixels)
    temp = [f'perc_invalid: {avg_perc_invalid:6.3f} %', '\n']
    f.writelines(temp)

    #calculate mean & std-------------------------------------------------------------
    mean /= 100 #divide by 100 to get %
    std = 0
    x = np.arange(0.02, 1.02, 0.02)

    for i, prob in zip(x, all_prob):
        std += pow((i-mean),2)*(prob/100)
    std = pow(std, 0.5)
    temp = [f'mean: {mean:.3f}', '\n', f'standard deviation of the probability distribution: {std:.3f}', '\n']
    f.writelines(temp)
    return

#takes in an image, use single view depth prediction network to predict the depth, then calls classify() to classify the img
#returns the standard deviation of the img, class of the image(0 for indoor 1 for outdoor), and the depth prediction tensor
def save_outputs(img_path, output_file_name):
    class_result = 0
    with torch.no_grad():
        save_path = os.path.join(output_path, f'{output_file_name}_depth.png')

        print(f'Reading input {img_path} ...')
        img = Image.open(img_path)

        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

        rgb_path = os.path.join(output_path, f'{output_file_name}_rgb.png')
        trans_rgb(img).save(rgb_path)

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3,1)

        output = model(img_tensor).clamp(min=0, max=1)

        std, class_result = classify(output.cpu().numpy())

        # output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
        # output = output.clamp(0,1)
        # output = 1 - output
        # plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')
        
        # print(f'Writing output {save_path} ...')
        print(f'classification result: {class_result}')
    return std, class_result, output


#classifies an image based on its depth prediction, this is called by save_outputs()
#returns the standard deviation of the img, and the classification (0 for indoor and 1 for outdoor)
def classify(depth_preds):

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
        #print(f'Range {min_depth:.2f} - {max_depth:.2f} | perc valid: {avg_perc_valid:6.3f} %')

        res = [f'{avg_perc_valid:6.3f}', '\n']
        f.writelines(res)

        #calculate mean
        mean += max_depth*avg_perc_valid

    # calculate percentage of invalid pixels------------------------------------------
    invalid_pixels = (depth_preds == 0).astype(np.float32)
    n_invalid_pixels = np.sum(invalid_pixels, axis=(1,2))
    perc_invalid_pixels = (n_invalid_pixels / n_total_pixels) * 100
    avg_perc_invalid = np.mean(perc_invalid_pixels)
    # print(f'perc_invalid: {avg_perc_invalid:6.3f} %')

    #calculate mean & std-------------------------------------------------------------
    mean /= 100 #divide by 100 to get %
    std = 0
    x = np.arange(0.02, 1.02, 0.02)

    for i, prob in zip(x, all_prob):
        std += pow((i-mean),2)*(prob/100)
    std = pow(std, 0.5)
    # print(f'mean: {mean:.3f}')
    print(f'standard deviation of the probability distribution: {std:.3f}')

    #classifies
    if (std > threshold):
        return std, 1
    else:
        return std, 0

#used to parse a file into an array, data separated by space
def parse_file(filename):
    with open(filename, "r") as f:
        data = [[str(y) for y in x.strip().split(" ")] for x in f]
        data[0] = [int(x) for x in data[0]]

        return data

#---------------------------------------------------------------------------------
# MAIN FUNCTION 
#---------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

# parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
# parser.set_defaults(im_name='NONE')

# parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
# parser.set_defaults(store_name='NONE')

parser.add_argument('--testing_file', dest='testing_file', help="path to file containing testing data")
parser.set_defaults(test_name='NONE')

parser.add_argument('--threshold', dest='threshold', help="path to file containing testing data")
parser.set_defaults(test_name='NONE')

args = parser.parse_args()

data = parse_file(args.testing_file)

root_dir = './pretrained_models/'

trans_topil = transforms.ToPILImage()

os.system(f"mkdir -p {output_path}")
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


results = []
threshold = float(args.threshold)

#get total number of indoor and outdoor data 
num_indoor = data[0][0]
num_outdoor = data[0][1]

#initialize variables used to calculate performance metrics
#note: outdoor output will be the positive class here 
true_pos = 0
false_pos = 0
failed_indoor_images = list()
failed_outdoor_images = list()
failed_indoor_std = 0
failed_outdoor_std = 0

for i in range(1, len(data)):
    img_path = Path(data[i][0])
    true_class = int(data[i][1])
    pred_class = 0
    
    res = []
    res.append(img_path)

    if img_path.is_file():
        std, pred_class, depths = save_outputs(img_path, os.path.splitext(os.path.basename(img_path))[0])
        res.append(std)
        res.append(pred_class)


    #compare classification results
    if (pred_class == 1 and true_class == 1):
        true_pos += 1
    elif (pred_class == 1 and true_class == 0):
        false_pos += 1
        temp = [img_path, depths]
        failed_indoor_images.append(temp)
        failed_indoor_std += std
    elif (pred_class == 0 and true_class == 1):
        temp = [img_path, depths]
        failed_outdoor_images.append(temp)
        failed_outdoor_std += std
    
    results.append(res)

failed_indoor_std /= len(failed_indoor_images) if (len(failed_indoor_images) != 0) else -1
failed_outdoor_std /= len(failed_outdoor_images) if (len(failed_outdoor_images) != 0) else -1

#get false negative and true negative
false_neg = num_outdoor - true_pos
true_neg = num_indoor - false_pos

#calculate performance metrics
true_pos_rate = true_pos/num_outdoor if num_outdoor != 0 else -1
false_pos_rate = false_pos/num_indoor if num_indoor != 0 else -1
true_neg_rate = true_neg/num_indoor if num_indoor != 0 else -1
false_neg_rate = false_neg/num_outdoor if num_outdoor != 0 else -1
# accuracy = (true_pos + true_neg)/(num_outdoor+num_indoor)

accuracy = (num_outdoor/(num_indoor+num_outdoor))*true_pos_rate +(num_indoor/(num_indoor+num_outdoor))*true_neg_rate #weighted
error_rate = (false_pos + false_neg)/(num_outdoor+num_indoor)
precision = true_pos/(true_pos + false_pos)

#create directories and folders needed to save results
temp = args.testing_file.split("/")
res_summary = f'summary'
filepath = f'/data/ersp21/classifier/results/{temp[-1]}/t={threshold}'
failed_indoor_dir = filepath + "/failed_indoor"
failed_outdoor_dir = filepath + "/failed_outdoor"

if not os.path.exists(filepath):
    os.makedirs(filepath)
if not os.path.exists(failed_indoor_dir):
    os.makedirs(failed_indoor_dir)
if not os.path.exists(failed_outdoor_dir):
    os.makedirs(failed_outdoor_dir)

filepath = os.path.join(filepath, res_summary)
f = open(filepath, "w")

f.writelines("-------- all image paths, their std, and prediction (1 for outdoor and 0 for indoor) ------------\n")
for res in results:
    if (len(res) == 3):
        f.writelines(f'image path: {res[0].absolute().as_posix()} | std: {res[1]:5f} | classification: {res[2]}\n')
    else:
        f.writelines(f'image path: {res[0].absolute().as_posix()} is invalid\n')

f.writelines("----------------------- incorrectly classified images --------------------\n")
f.writelines("indoor:\n")
for path in failed_indoor_images:
    f.writelines(f'{path[0]}\n')
    shutil.copy2(path[0], failed_indoor_dir)
    
    img_name = os.path.splitext(os.path.basename(path[0]))[0] #get name of the img
    save_path = os.path.join(failed_indoor_dir, f'{img_name}_depth.png') #create path to save the depth of the img

    #save depth distributions in textfile
    save_depthDistributions(path[1].cpu().numpy(), os.path.join(failed_indoor_dir, f'{img_name}_depthDist.txt')) 

    #change predicted depth format so we can save the depth map
    output = F.interpolate(path[1].unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
    output = output.clamp(0,1)
    output = 1 - output

    #save depth map
    plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')
f.writelines(f'average standard deviation: {failed_indoor_std:.4f}\n')
f.writelines("outdoor:\n")
for path in failed_outdoor_images:
    f.writelines(f'{path[0]}\n')
    shutil.copy2(path[0], failed_outdoor_dir)

    img_name = os.path.splitext(os.path.basename(path[0]))[0]
    save_path = os.path.join(failed_outdoor_dir, f'{img_name}_depth.png')

    save_depthDistributions(path[1].cpu().numpy(), os.path.join(failed_outdoor_dir, f'{img_name}_depthDist.txt'))

    output = F.interpolate(path[1].unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
    output = output.clamp(0,1)
    output = 1 - output

    plt.imsave(save_path, path[1].detach().cpu().squeeze(),cmap='viridis')
f.writelines(f'average standard deviation: {failed_outdoor_std:.4f}\n')

f.writelines(f'------------ Performance w/ threshold={threshold}---------------------\n')
f.writelines(f'true positive rate: {true_pos_rate}\n')
f.writelines(f'false positive rate: {false_pos_rate}\n')
f.writelines(f'accuracy: {accuracy}\n')
f.writelines(f'error rate: {error_rate}\n')
f.writelines(f'precision: {precision}\n')


