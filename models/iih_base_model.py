import torch
import os
import itertools
import torch.nn.functional as F
from util import distributed as du
from .base_model import BaseModel
from util import util
from . import harmony_networks as networks
import torch.nn as nn
from torchvision import models
from itertools import chain

class IIHBaseModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='base', dataset_mode='ihd')
        if is_train:
            parser.add_argument('--coef_bg', type=float, default=100., help='weight for L1 loss')
            parser.add_argument('--coef_fg', type=float, default=100., help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.loss_names = ['G', 'G_bg', 'G_fg']            
        self.visual_names = ['mask', 'comp', 'real', 'out_fg', 'out_bg']
        self.test_visual_names = ['mask', 'comp', 'out_fg', 'out_bg', 'real']
        self.model_names = ['G']
        self.opt.device = self.device
        self.netG = networks.define_G(opt.norm, opt.netG, opt.init_type, opt.init_gain, opt)
        self.cur_device = torch.cuda.current_device()
        self.ismaster = du.is_master_proc(opt.NUM_GPUS)
        if self.ismaster:
            print(self.netG)  
        if self.isTrain:
            util.saveprint(self.opt, 'netG', str(self.netG))  
            self.criterionL1 = torch.nn.L1Loss().cuda(self.cur_device)
            self.criterionL2 = torch.nn.MSELoss().cuda(self.cur_device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)


    def set_input(self, input):
        self.comp = input['comp'].to(self.device)
        self.real = input['real'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.image_paths = input['img_path']
        
    def forward(self):
        self.out_fg, self.out_bg = self.netG(self.comp, self.mask)           

    def backward(self):
        self.loss_G_bg = self.criterionL1(self.out_bg, self.real) * self.opt.coef_bg
        self.loss_G_fg = self.criterionL1(self.out_fg, self.comp) * self.opt.coef_fg
        self.loss_G = self.loss_G_fg + self.loss_G_bg
        self.loss_G.backward()

        
    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


    def gradient_loss(self, input_1, input_2):
        g_x = self.criterionL1(util.gradient(input_1, 'x'), util.gradient(input_2, 'x'))
        g_y = self.criterionL1(util.gradient(input_1, 'y'), util.gradient(input_2, 'y'))
        return g_x+g_y