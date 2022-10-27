from glob import glob0
from gzip import BadGzipFile
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.optim import lr_scheduler
from torchvision import models
from util.tools import *
from util import util
from . import networks as networks_init
import torchvision.transforms.functional as tff
import math


###############################################################################
# Helper Functions
###############################################################################

def normalize(v):
    if type(v) == list:
        return [normalize(vv) for vv in v]

    return v * torch.rsqrt((torch.sum(v ** 2, dim=1, keepdim=True) + 1e-8))

def exists(val):
    return val is not None

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(norm='instance', netG='base', init_type='normal', init_gain=0.02, opt=None):
    """Create a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    
    if netG == 'base':
        net = BaseGenerator(opt)
        
    net = networks_init.init_weights(net, init_type, init_gain)
    net = networks_init.build_model(opt, net)

    return net

class BaseGenerator(nn.Module):
    def __init__(self, opt):
        super(BaseGenerator, self).__init__()
        self.device = opt.device
        self.reflectance_dim = 512
        self.encoder = ContentEncoder(opt.input_nc, self.reflectance_dim, opt.ngf, 'in', opt.activ, opt.pad_type,
                                      opt.spatial_code_ch, opt.global_code_ch)
        self.generator = ContentDecoder(self.encoder.output_dim, opt.output_nc, opt.ngf, opt.pad_type,
                                        opt.spatial_code_ch, opt.global_code_ch)

        
    def forward(self, img, mask):
        sp, fgl, bgl = self.encoder(img, mask)

        from_fg = self.generator(sp, fgl)
        from_fg = img*(1-mask) + from_fg*mask
        
        from_bg = self.generator(sp, bgl)
        from_bg = img*(1-mask) + from_bg*mask
        
        return from_fg, from_bg

      
##################################################################################
# Encoder and Decoders
##################################################################################

class ContentEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, dim, norm, activ, pad_type, spatial_code_ch, global_code_ch):
        super(ContentEncoder, self).__init__()

        self.down0 = Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)
        self.down1 = ResBlock(dim, dim * 2, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.down2 = ResBlock(dim * 2, dim * 4, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.down3 = ResBlock(dim * 4, dim * 8, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        
        self.ToFGCode = nn.Sequential(
                Conv2dBlock(dim * 8, dim * 8, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type), #16
                Conv2dBlock(dim * 8, dim * 8, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type), #8
                Conv2dBlock(dim * 8, dim * 8, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type), #4
        )
        
        self.ToBGCode = nn.Sequential(
                Conv2dBlock(dim * 8, dim * 8, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type), #16
                Conv2dBlock(dim * 8, dim * 8, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type), #8
                Conv2dBlock(dim * 8, dim * 8, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type), #4
        )
        
        self.ToSpatialCode = nn.Sequential(
                Conv2dBlock(output_dim, spatial_code_ch, 1, 1, 0, norm=norm, activation=activ, pad_type=pad_type),
        )

        self.output_dim = output_dim
        
        self.fg_mlp = MLP(global_code_ch, global_code_ch, 8)
        self.bg_mlp = MLP(global_code_ch, global_code_ch, 8)
        
    def forward(self, img, mask):
        x = torch.cat((img, mask), dim=1)
        
        x = self.down0(x)        
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        
        sp = self.ToSpatialCode(x) # torch.Size([8, 8, 32, 32])
        fgl = self.ToFGCode(x)
        bgl = self.ToBGCode(x)
        
        fgl = self.fg_mlp(fgl.mean(dim=(2,3))) #fg.mean(dim=(2,3))
        bgl = self.bg_mlp(bgl.mean(dim=(2,3)))
                
        return sp, fgl, bgl #gl
    
    def extract_stats(self, feature, mask):
        down_mask = F.interpolate(mask, size=feature.size(2))
        ft = down_mask*feature
        
        gap = F.adaptive_avg_pool2d(ft, 1)
        gap = gap.squeeze(3).squeeze(2)
    
        return gap

class ContentDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, dim, pad_type, spatial_code_ch, global_code_ch):
        super(ContentDecoder, self).__init__()
        
        dim = input_dim
        
        self.IntoFeature = GeneratorModulation(global_code_ch, spatial_code_ch)
                
        self.gen1 = GeneratorBlock(global_code_ch, dim, dim, upsample = False)
        self.gen2 = GeneratorBlock(global_code_ch, dim, dim//2, upsample = True)
        self.gen3 = GeneratorBlock(global_code_ch, dim//2, dim // 4, upsample = True)
        self.gen4 = GeneratorBlock(global_code_ch, dim // 4, dim // 8, upsample = True)
        self.out = Conv2dBlock(dim // 8, output_dim, 3, 1, 1, norm='none', activation='tanh', pad_type=pad_type)
        

    def forward(self, sp, gl):
        x = self.IntoFeature(sp, gl) # torch.Size([8, 512, 32, 32])
        x = self.gen1(x, gl) # torch.Size([8, 512, 32, 32])
        x = self.gen2(x, gl) # torch.Size([8, 256, 64, 64])
        x = self.gen3(x, gl) # torch.Size([8, 128, 128, 128])
        x = self.gen4(x, gl) # torch.Size([8, 64, 256, 256])
        x = self.out(x) #torch.Size([8, 3, 256, 256])
        
        return x

##################################################################################
# Basic Blocks
##################################################################################

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, input_dim//2, norm=norm, activation=activ)]
        dim = input_dim//2
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim//2, norm=norm, activation=activ)]
            dim = dim//2
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
    
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, s=1, p=1, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(in_dim, in_dim, k, s, p, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(in_dim, out_dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)
        
        self.skip = Conv2dBlock(in_dim, out_dim, k, s, p, norm=norm, activation='none', pad_type=pad_type)

    def forward(self, x):
        residual = self.skip(x)
        out = self.model(x)
        out = (out+residual) / math.sqrt(2)
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        self.norm_type = norm
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'grp':
            self.norm = nn.GroupNorm(groupcount, norm_dim)
        
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(ConvTranspose2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'grp':
            self.norm = nn.GroupNorm(groupcount, norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=self.use_bias))
        else:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# Normalization layers
##################################################################################
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):  
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info

#############################################
################
# https://github.com/gaussian37/pytorch_deep_learning_models/blob/master/u-net/u_net.py
################

def ConvBlock(in_dim, out_dim, act_fn, norm, kernel_size=3, stride=1, padding=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = padding),
        norm(out_dim),
        act_fn,
    )
    return model

def ConvTransBlock(in_dim, out_dim, act_fn, norm, kernel_size=3, stride=1, padding=1):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = padding),
        norm(out_dim),
        act_fn,
    )
    return model

def Maxpool():
    pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    return pool

def ConvBlock2X(in_dim, out_dim, act_fn, norm, kernel_size=3, stride=1, padding=1):
    model = nn.Sequential(
        ConvBlock(in_dim, out_dim, act_fn, norm, kernel_size),
        ConvBlock(out_dim, out_dim, act_fn, norm, kernel_size),
    )
    return model


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), nn.LeakyReLU(0.2)])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)

class GeneratorModulation(torch.nn.Module):
    def __init__(self, styledim, outch):
        super().__init__()
        self.scale = EqualLinear(styledim, outch)
        self.bias = EqualLinear(styledim, outch)

    def forward(self, x, style):
        if style.ndimension() <= 2:
            return x * (1 * self.scale(style)[:, :, None, None]) + self.bias(style)[:, :, None, None]
        else:
            style = F.interpolate(style, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            return x * (1 * self.scale(style)) + self.bias(style)
        
class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, istyle): #inoise
        if exists(self.upsample):
            x = self.upsample(x)
            
        style1 = self.to_style1(istyle) #torch.Size([8, 8])
        
        x = self.conv1(x, style1)
        x = self.activation(x)
        
        style2 = self.to_style2(istyle) #torch.Size([8, 256])
        x = self.conv2(x, style2) #
        x = self.activation(x)
        
        return x
    
class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y): # x:spatial y:style
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x