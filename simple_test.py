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
    # checkpoint = torch.load('./checkpoint/edge2color/ckpt.pth')
    checkpoint = torch.load('./checkpoints/iih_base_allihd_test/latest_net_G.pth', map_location=opt.device)
    #current_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint, strict=False)
    # model.load_state_dict(checkpoint['Sketch2Color'], strict=True)
    # netD.load_state_dict(checkpoint['netD'], strict=True)
    # loss_list_D = checkpoint['loss_list_D'],
    # loss_list_G = checkpoint['loss_list_G'],
    # optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    # optimizer_D.load_state_dict(checkpoint['optimizer_D']),
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
    parser.add_argument('--dataset_root', type=str, default="//home/ubuntu/IHD/")
    # parser.add_argument('--dataset_name', type=str, default="IHD")
    parser.add_argument('--evaluation_type', type=str, default="our")
    parser.add_argument('--dataset_name', type=str, default="IHD")
    parser.add_argument('--result_root', type=str, default="results/iih_base_allihd_test/test_latest/images/")
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

    # m = torch.nn.Upsample(scale_factor=16)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'



    model = load(model)
    model.eval()

    test_imagefolder = ihd_dataset.IhdDataset(opt=opt)
    test_loader = torch.utils.data.DataLoader(test_imagefolder, batch_size=opt.batch_size, shuffle=True)
    print("Done!")
    print("Test data size : {}".format(len(test_imagefolder)))
    test_batch = next(iter(test_loader))
    # temp_batch_iter = iter(train_loader)

    with torch.no_grad():
        comp = test_batch['comp']
        real = test_batch['real']
        mask = test_batch['mask']
        image_paths = test_batch['img_path']
            
        sp, fgl, bgl = model.encoder(comp, mask)
        
        out_f = model.generator(sp, fgl)
        out_f = comp*(1-mask) + out_f*mask
        
        out_b = model.generator(sp, bgl)
        out_b = comp*(1-mask) + out_b*mask

        zero = torch.zeros_like(fgl)
        out_z = model.generator(sp, zero)
        out_z = comp*(1-mask) + out_z*mask
        
        one = torch.ones_like(fgl)
        out_o = model.generator(sp, one)
        out_o = comp*(1-mask) + out_o*mask
        
        num_samples = 1
        Din = 512
        normal = torch.distributions.normal.Normal(loc=0, scale=1).sample((num_samples, Din))
        out_n = model.generator(sp, normal)
        out_n = comp*(1-mask) + out_n*mask

        plt.figure(figsize=(16, 16))
        result = torch.cat((comp, real, out_f, out_b, out_z, out_o, out_n), dim=-1)
        # result = np.transpose(vutils.make_grid(result, nrow=1, padding=5).cpu().numpy(), (1, 2, 0))
        result = util.tensor2im(result)
        plt.imshow(result)
        plt.axis("off")
        plt.title("Comp / Real / fg / bg / zero / one / normal", fontsize=10)
        plt.imsave('out5.png', result)


        # # """Now let's visualize them"""
        # #
        # # # get the feature map shape
        # # h, w = c_conv_features['0'].tensors.shape[-2:]
        # #
        # # fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
        # # colors = COLORS * 100
        # # for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
        # #     ax = ax_i[0]
        # #     ax.imshow(dec_attn_weights[0, idx].view(h, w))
        # #     ax.axis('off')
        # #     ax.set_title(f'query id: {idx.item()}')
        # #     ax = ax_i[1]
        # #     ax.imshow(im)
        # #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        # #                                fill=False, color='blue', linewidth=3))
        # #     ax.axis('off')
        # #     ax.set_title(CLASSES[probas[idx].argmax()])
        # # fig.tight_layout()

        # # output of the CNN
        # f_map_e = e_conv_features # torch.Size([1, 992, 16, 16])
        # f_map_c = c_conv_features # torch.Size([1, 992, 16, 16])
        # print("Encoder attention:      ", enc_attn_weights[0].shape)
        # print("Feature map:            ", f_map_e.shape)
        # print("Feature map:            ", f_map_c.shape)

        # # get the HxW shape of the feature maps of the CNN
        # # shape = f_map_e.shape[-2:]
        # # and reshape the self-attention to a more interpretable shape
        # sattn = enc_attn_weights[0].reshape(shape + shape)
        # print("Reshaped self-attention:", sattn.shape)

        # # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
        # fact = 16

        # # let's select 4 reference points for visualization
        # idxs = [(64, 64), (64, 128), (128, 128), (192, 128), ]

        # # here we create the canvas
        # fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
        # # and we add one plot per reference point
        # gs = fig.add_gridspec(2, 4)
        # axs = [
        #     fig.add_subplot(gs[0, 0]),
        #     fig.add_subplot(gs[1, 0]),
        #     fig.add_subplot(gs[0, -1]),
        #     fig.add_subplot(gs[1, -1]),
        # ]

        # # for each one of the reference points, let's plot the self-attention
        # # for that point
        # for idx_o, ax in zip(idxs, axs):
        #     idx = (idx_o[0] // fact, idx_o[1] // fact)
        #     ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
        #     ax.axis('off')
        #     ax.set_title(f'self-attention{idx_o}')

        # # and now let's add the central image, with the reference points as red circles
        # fcenter_ax = fig.add_subplot(gs[:, 1:-1])
        # fcenter_ax.imshow(edge.squeeze())
        # for (y, x) in idxs:
        #     # scale = edge.height / edge.shape[-2]
        #     # x = ((x // fact) + 0.5) * fact
        #     # y = ((y // fact) + 0.5) * fact
        #     # fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
        #     fcenter_ax.add_patch(plt.Circle((x, y), fact // 2, color='r'))
        #     fcenter_ax.axis('off')
        # plt.show()