"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os.path
import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from util import util
from skimage import color
import cv2
import math

class IhdDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.image_paths = []
        self.isTrain = opt.isTrain
        self.image_size = opt.crop_size
        
        if opt.isTrain==True:
            print('loading training file')
            self.trainfile = opt.dataset_root+opt.dataset_name+'_train.txt'
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(os.path.join(opt.dataset_root,'composite_images',line.rstrip()))
        elif opt.isTrain==False:
            #self.real_ext='.jpg'
            print('loading test file')
            # self.trainfile = opt.dataset_root+opt.dataset_name+'_test.txt'
            self.trainfile = opt.dataset_root+opt.dataset_name+'_test.txt'
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(os.path.join(opt.dataset_root,'composite_images',line.rstrip()))
        # get the image paths of your dataset;
          # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        transform_list = [
            transforms.ToTensor(),
            # transforms.Normalize((0, 0, 0), (1, 1, 1))
        ]
        self.transforms = transforms.Compose(transform_list)
        self.color_transforms = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5) #brightness=(0,1), contrast=(0,1), hue=(0,1)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = self.image_paths[index]
        name_parts=path.split('_')
        mask_path = self.image_paths[index].replace('composite_images','masks')
        mask_path = mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
        target_path = self.image_paths[index].replace('composite_images','real_images')
        target_path = target_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')

        comp = Image.open(path).convert('RGB')
        real = Image.open(target_path).convert('RGB')
        mask = Image.open(mask_path).convert('1')

        if np.random.rand() > 0.5 and self.isTrain:
            comp, mask, real = tf.hflip(comp), tf.hflip(mask), tf.hflip(real)
            
        if comp.size[0] != self.image_size:
            comp = tf.resize(comp, [self.image_size, self.image_size])
            mask = tf.resize(mask, [self.image_size, self.image_size])
            real = tf.resize(real, [self.image_size, self.image_size])
                    
        if self.isTrain:
            colored = self.color_transforms(comp)
        
        comp = self.transforms(comp)
        mask = tf.to_tensor(mask)
        real = self.transforms(real)
        
        if self.isTrain:
            colored = self.transforms(colored)
            comp = colored*mask + comp*(1-mask)

        return {'comp': comp, 'real': real, 'mask':mask, 'img_path':path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
    
    def get_affine_matrix(self, center, angle, translate, scale, shear):

        rot = math.radians(angle)
        sx = math.radians(shear[0])
        sy = math.radians(shear[1])

        cx, cy = center
        tx, ty = translate

        # RSS without scaling
        a = math.cos(rot - sy) / math.cos(sy)
        b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
        c = math.sin(rot - sy) / math.cos(sy)
        d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)
        
        # #################
        aff = [a, b, 0.0, c, d, 0.0]
        aff = [x * scale for x in aff]
        
        aff[2] += (cx + tx)
        aff[5] += (cy + ty)
        
        aff[2] -= cx
        aff[5] -= cy

        aff[2] /= 255 # image_size-1, normalize into (-1, 1)
        aff[5] /= 255
        # #################

        return np.array(aff)