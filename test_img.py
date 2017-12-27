# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.image
import cv2

# parse parameters
parser = argparse.ArgumentParser(description='Attributes swapping')
parser.add_argument("--model_path", type=str, default="models/best_rec_ae.pth",
                    help="Trained model path")
parser.add_argument("--img_path", type=str, default="data/mytest",
                    help="Trained model path")
parser.add_argument("--n_interpolations", type=int, default=2,
                    help="Number of interpolations per image")
parser.add_argument("--alpha_min", type=float, default=0,
                    help="Min interpolation value")
parser.add_argument("--alpha_max", type=float, default=1,
                    help="Max interpolation value")
parser.add_argument("--output_path", type=str, default="results/smiling_male.png",
                    help="Output path")
params = parser.parse_args()

# load trained model
ae = torch.load(params.model_path).eval()

# restore main parameters
params.attr = ['Smiling', 'Male']
params.n_attr = len(params.attr)

""" load data """
def read_imgs():
    imgs = []
    for filename in os.listdir(params.img_path):
        img = cv2.imread(params.img_path + '/' + filename)
        # resize
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        img = img[np.newaxis,:]
        img = torch.from_numpy(img)
        # normalization
        img = img.float().div_(255.0).mul_(2.0).add_(-1)  # [-1, 1]
        img = img.permute(0, 3, 1, 2)  # [1, 256,256,3] -> [1, 3, 256,256]
        imgs.append(img)
    imgs = torch.cat(imgs, 0)
    return Variable(imgs.cuda(), volatile = True)

""" Reconstruct images / create interpolations """
def get_interpolations(ae, images, params):
    enc_output = ae.encode(images)

    # interpolation values
    alphas = np.linspace(params.alpha_min, params.alpha_max, params.n_interpolations)
    alphas = [torch.FloatTensor([[1 - alpha, alpha],[1 - alpha, alpha]]) for alpha in alphas]

    # original / interpolations
    outputs = []
    outputs.append(images)
    #outputs.append(ae.decode(enc_outputs, attributes)[-1])
    for alpha in alphas:
        alpha = Variable(alpha.cuda())
        outputs.append(ae.decode(enc_output, alpha))

    # return stacked images
    return torch.cat([x.unsqueeze(1) for x in outputs], 1).data.cpu() # [img_nums, n_interpolations, 3, 256, 256]


imgs = read_imgs()
interpolations = get_interpolations(ae, imgs, params)

""" Create a grid with all images. """
def get_grid(images):
    n_images, n_columns, img_fm, img_sz, _ = images.size()
    images = images.view(n_images * n_columns, img_fm, img_sz, img_sz) # [img_nums * n_interpolations, 3, 256, 256]
    images.add_(1).div_(2.0) # dec 最后一层为tanh, 因此需转化为0-1
    return make_grid(images, nrow=n_columns) # nrow 每行显示几张图片

# generate the grid / save it to a PNG file
grid = get_grid(interpolations) # [3, 1034, 776]
matplotlib.image.imsave(params.output_path, grid.numpy().transpose((1, 2, 0)))
