import math

from PIL import Image
import requests
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

import os
from models import harmony_networks
from data import ihd_dataset
import numpy as np
import IPython
import argparse
import torchvision.utils as vutils
from util import util


torch.set_grad_enabled(False)

# if device=='cuda':
#     print("The gpu to be used : {}".format(torch.cuda.get_device_name(0)))
# else:
#     print("No gpu detected")

def load(model):
    # global current_epoch, best_losses, loss_list_D, loss_list_G, optimizer_G, optimizer_D
    print('Loading...', end=' ')
    assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoints/iih_base_allihd_test/latest_net_G.pth', map_location=opt.device)
    model.load_state_dict(checkpoint, strict=False)

    return model

# configs ######################################
def get_args_parser():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--input_nc', type=int, default=4, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--netG', type=str, default='base', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_downsample', type=int, default=3, help='min 2')
    parser.add_argument('--enc_n_res', type=int, default=0, help='reflectance encoder resblock layers')
    parser.add_argument('--dec_n_res', type=int, default=0, help='reflectance decoder resblock layers')
    parser.add_argument('--pad_type', type=str, default='reflect', help='pad_type')
    parser.add_argument('--activ', type=str, default='lrelu', help='activ')
    parser.add_argument("--reflect_dim", default=512, type=int)
    parser.add_argument("--spatial_code_ch", default=512, type=int)
    parser.add_argument("--global_code_ch", default=512, type=int)
    parser.add_argument("--isTrain", default=False, type=bool)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--model', type=str, default='iih_base')
    # parser.add_argument('--dataset_root', type=str, default="//home/ubuntu/IHD/")
    parser.add_argument('--dataset_root', type=str, default="/home/ubuntu/RealIndoors/")
    # parser.add_argument('--dataset_name', type=str, default="IHD")
    parser.add_argument('--evaluation_type', type=str, default="our")
    # parser.add_argument('--dataset_name', type=str, default="IHD")
    parser.add_argument('--dataset_name', type=str, default="RealIndoors")
    parser.add_argument('--result_root', type=str, default="results/iih_base_allihd_test/test_latest/latentspace/")
    parser.add_argument('--dataset_mode', type=str, default='ihd', help='chooses how datasets are loaded.')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--preprocess', type=str, default='none', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')

    return parser

if __name__ == '__main__':
    opt = get_args_parser().parse_args()

    model = harmony_networks.BaseGenerator(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf, norm=opt.norm, opt=opt)
    optimizer_G = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(opt.beta1, opt.beta2))

    model = load(model)
    model.eval()

    test_imagefolder = ihd_dataset.IhdDataset(opt=opt)
    test_loader = torch.utils.data.DataLoader(test_imagefolder, batch_size=opt.batch_size, shuffle=False)
    print("Done!")
    print("Test data size : {}".format(len(test_imagefolder)))


    for i, data in enumerate(test_loader):
        outputs = []
            
        with torch.no_grad():
            comp = data['comp']
            real = data['real']
            mask = data['mask']
            image_paths = data['img_path']
            
            save_name = image_paths[0].split('/')[-1]
            save_path = os.path.join(opt.result_root, save_name)
                
            sp, fgl, bgl = model.encoder(comp, mask)
            
            normal = torch.normal(0, 0.2, size=(1,512))
            normal += fgl
            
            alphas = torch.linspace(0, 1, steps=5)

            for a in alphas:
                latent = a*fgl + (1-a)*normal
                out = model.generator(sp, latent)
                out = comp*(1-mask) + out*mask
                outputs.append(out)
                
            normal = torch.normal(0, 0.1, size=(1,512))
            out = model.generator(sp, normal)
            out = comp*(1-mask) + out*mask
            outputs.append(out)
            
            outputs = torch.cat(outputs, dim=-1)
            
            mask = torch.cat((mask, mask, mask), dim=1)

            plt.figure(figsize=(16, 16))
            result = torch.cat((mask, comp, outputs), dim=-1)
            result = util.tensor2im(result)
            plt.axis("off")
            plt.title("Mask / Comp / 0:100 / 25:75 / 50:50 / 75:25 / 100:0/ Random", fontsize=10)
            plt.imsave(save_path, result)