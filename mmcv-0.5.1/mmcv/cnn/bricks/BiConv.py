# coding=utf-8
from __future__ import division
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
import collections
from itertools import repeat
layer = 0
ks1 = 3

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class Binarization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, scale):

        scale = torch.abs(scale)
        
        bin = 0.02
        
        weight_bin = torch.sign(weight) * bin

        output = weight_bin * scale

        ctx.save_for_backward(weight, scale)
        
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        weight, scale = ctx.saved_tensors
        
        para_loss = 0.0001
        bin = 0.02
        weight_bin = torch.sign(weight) * bin
        
        gradweight = para_loss * (weight - weight_bin * scale) + (gradOutput * scale)
        
        #pdb.set_trace()
        grad_scale_1 = torch.sum(torch.sum(torch.sum(gradOutput * weight,keepdim=True,dim=3),keepdim=True, dim=2),keepdim=True, dim=1)
        
        grad_scale_2 = torch.sum(torch.sum(torch.sum((weight - weight_bin * scale) * weight_bin,keepdim=True,dim=3),keepdim=True, dim=2),keepdim=True, dim=1)

        gradMFilter = grad_scale_1 - para_loss * grad_scale_2
        #pdb.set_trace()
        return gradweight, gradMFilter

class BiConv(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BiConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')

        self.generate_scale()
        self.binarization = Binarization.apply
        self.binfunc = BinaryActivation()
        self.out_channels = out_channels
        
    def generate_scale(self):
        self.scale = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def forward(self, x):

        x = self.binfunc(x)

        new_weight = self.binarization(self.weight, self.scale)
             
        return F.conv2d(x, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
class BiRConv(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BiRConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')

        self.generate_scale()
        self.binarization = Binarization.apply
        self.binfunc = BinaryActivation()
        self.out_channels = out_channels
        
    def generate_scale(self):
        self.scale = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def forward(self, x):
        xi = x
        xi = self.binfunc(xi)
        new_weight = self.binarization(self.weight, self.scale)
        conv1 = F.conv2d(xi, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
        x = self.binfunc(x)
        new_weight2 = self.binarization(self.weight, self.scale)
        conv2 = F.conv2d(x, new_weight2, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return conv1 + conv2
    
class JLBPwBiConv(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(JLBPwBiConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')

        self.generate_scale()
        self.register_buffer('_pre_kernel', self.get_kernel()) # 24*C_in, 1, 5, 5
        self.binarization = Binarization.apply
        self.binfunc = BinaryActivation()
        self.out_channels = out_channels
        self.shape = (out_channels, 24 * in_channels // 1, 1, 1)
        self.weights = nn.Parameter(torch.Tensor(*self.shape))
        
    def generate_scale(self):
        self.scale = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def get_kernel(self):
        kernel = torch.tensor([ [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
        '''
        kernel = torch.tensor([ [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])'''
        #JLBP1 是中间一列全为-1
        kernel = kernel.view(24, 1, 5, 5)
        return kernel.contiguous().repeat(self.in_channels, 1, 1, 1)
    
    def forward(self, x):
        #print(x.shape)
        #print(self.pre_kernel.shape)
        xi = x
        x = F.conv2d(x, self.pre_kernel, stride=1, padding=2, 
                groups=self.in_channels, dilation=1) # b, 8*C_in, H, W
        #x = self.sign_layer(x)
        mask = torch.ones(x.size()[2:], device=x.device) * 1e-8
        mask[::2, :] *= -1
        mask[:, ::2] *= -1
        x = self.binfunc(x + mask.expand_as(x))

        new_weight = self.binarization(self.weights, self.scale)

        #return F.conv2d(x, new_weight, self.bias, self.stride,
        #                self.padding, self.dilation, self.groups)
        conv1 = F.conv2d(x, new_weight, stride=1, padding=0,
                groups=1)
        
        '''xi = self.binfunc(xi)

        new_weight = self.binarization(self.weight, self.scale)
             
        conv2 = F.conv2d(xi, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)'''
        return conv1
    
    @property
    def pre_kernel(self):
        return self._pre_kernel

class LBPConv(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(LBPConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')

        self.generate_scale()
        self.register_buffer('_pre_kernel', self.get_kernel()) # 8*C_in, 1, 3, 3
        self.binarization = Binarization.apply
        self.binfunc = BinaryActivation()
        self.out_channels = out_channels
        self.shape = (out_channels, 8 * in_channels // 1, 1, 1)
        self.weights = nn.Parameter(torch.Tensor(*self.shape))
        
    def generate_scale(self):
        self.scale = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def get_kernel(self):
        kernel = torch.zeros(8, 9)
        indices = torch.stack([torch.tensor([0, 1, 2, 3, 5, 6, 7, 8]).long(),
            torch.ones(8).long() * 4]).transpose(1, 0)
        values = torch.ones(8,2)
        values[:, 1] = -1
        kernel.scatter_(1, indices, values)
        kernel = kernel.view(8, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        return kernel.contiguous()
    
    def forward(self, x):
        #print(x.shape)
        #print(self.pre_kernel.shape)
        x = F.conv2d(x, self.pre_kernel, stride=1, padding=1, 
                groups=self.in_channels, dilation=1) # b, 8*C_in, H, W
        #x = self.sign_layer(x)
        mask = torch.ones(x.size()[2:], device=x.device) * 1e-8
        mask[::2, :] *= -1
        mask[:, ::2] *= -1
        x = self.binfunc(x + mask.expand_as(x))

        new_weight = self.binarization(self.weights, self.scale)

        #return F.conv2d(x, new_weight, self.bias, self.stride,
        #                self.padding, self.dilation, self.groups)
        return F.conv2d(x, new_weight, stride=1, padding=0,
                groups=1)
    
    @property
    def pre_kernel(self):
        return self._pre_kernel

class LBPConv2(_ConvNd): #13.5%
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(LBPConv2, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')

        self.generate_scale()
        self.register_buffer('_pre_kernel', self.get_kernel()) # 8*C_in, 1, 3, 3
        self.binarization = Binarization.apply
        self.binfunc = BinaryActivation()
        self.out_channels = out_channels
        self.shape = (out_channels, 8 * in_channels // groups, 1, 1)
        self.shape3 = (out_channels, in_channels // groups, 3, 3)
        self.shape5 = (out_channels, in_channels // groups, 5, 5)
        self.shape7 = (out_channels, in_channels // groups, 7, 7)
        self.shape9 = (out_channels, in_channels // groups, 9, 9)
        self.weights = nn.Parameter(torch.Tensor(*self.shape))
        self.weights3 = nn.Parameter(torch.Tensor(*self.shape3))
        self.weights5 = nn.Parameter(torch.Tensor(*self.shape5))
        self.weights7 = nn.Parameter(torch.Tensor(*self.shape7))
        self.weights9 = nn.Parameter(torch.Tensor(*self.shape9))
        
    def generate_scale(self):
        self.scale = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def get_kernel(self, scale_factor=2):
        '''kernel = torch.tensor([ [ 1.,  0.,  0.,  0.,  -1.,  0.,  0.,  0.,  0.],
                                    [ 0.,  1.,  0.,  0.,  -1.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  1.,  0.,  -1.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  1.,  -1.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0.,  -1.,  1.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0.,  -1.,  0.,  1.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0.,  -1.,  0.,  0.,  1.,  0.],
                                    [ 0.,  0.,  0.,  0.,  -1.,  0.,  0.,  0.,  1.]])
        kernel = kernel.view(8, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        if self.kernel_size == (5, 5):
            kernel = F.interpolate(kernel, size=(5, 5), mode='nearest')
            #print("5*5")
            #print(kernel)
        elif self.kernel_size == (7, 7):
            kernel = F.interpolate(kernel, size=(7, 7), mode='nearest')
        elif self.kernel_size == (9, 9):
            kernel = F.interpolate(kernel, size=(9, 9), mode='nearest')
        return kernel.contiguous()'''
        
        
        kernel = torch.tensor([ [ 1.,  0.,  0.,  0.,  -1.,  0.,  0.,  0.,  0.],
                                    [ 0.,  1.,  0.,  0.,  -1.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  1.,  0.,  -1.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  1.,  -1.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0.,  -1.,  1.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0.,  -1.,  0.,  1.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0.,  -1.,  0.,  0.,  1.,  0.],
                                    [ 0.,  0.,  0.,  0.,  -1.,  0.,  0.,  0.,  1.]])
        kernel = kernel.view(8, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        if self.kernel_size == (5, 5):
            kernel = torch.tensor([ [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
            kernel = kernel.view(8, 1, 5, 5).repeat(self.in_channels, 1, 1, 1)
        elif self.kernel_size == (7, 7):
            kernel = torch.tensor([ [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],#22
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],#28
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],#43
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],#46
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])#49
            kernel = kernel.view(8, 1, 7, 7).repeat(self.in_channels, 1, 1, 1)
        elif self.kernel_size == (9, 9):
            kernel = torch.tensor([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  -1.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                    [0., 0.,  0.,  0., 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  -1.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  -1.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  -1.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  -1.,  0.,
                                    0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  -1.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  -1.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  -1.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
            kernel = kernel.view(8, 1, 9, 9).repeat(self.in_channels, 1, 1, 1)
        return kernel.contiguous()
        
        '''
        kernel = torch.zeros(8, 9)
        indices = torch.stack([torch.tensor([0, 1, 2, 3, 5, 6, 7, 8]).long(),
            torch.ones(8).long() * 4]).transpose(1, 0)
        values = torch.ones(8,2)
        values[:, 1] = -1
        kernel.scatter_(1, indices, values)
        kernel = kernel.view(8, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        return kernel.contiguous()'''
    
    def forward(self, x):
        #print(x.shape)
        #print(self.pre_kernel.shape)
        xi = x
        ks = 3
        if self.kernel_size == (5, 5):
            ks = 5
        elif self.kernel_size == (7, 7):
            ks = 7
        elif self.kernel_size == (9, 9):
            ks = 9
        #print(ks)
        x = F.conv2d(x, self.pre_kernel, stride=1, padding=1 + (int)((ks-3)/2), 
                groups=self.in_channels, dilation=1) # b, 8*C_in, H, W
        #print(x.shape)
        #从这里可视化
        #x = self.sign_layer(x)
        mask = torch.ones(x.size()[2:], device=x.device) * 1e-8
        mask[::2, :] *= -1
        mask[:, ::2] *= -1
        x = self.binfunc(x + mask.expand_as(x))

        new_weight = self.binarization(self.weights, self.scale)
        #print(new_weight.shape)

        #return F.conv2d(x, new_weight, self.bias, self.stride,
        #                self.padding, self.dilation, self.groups)
        conv1 = F.conv2d(x, new_weight, stride=1, padding=0,
                groups=1)
        
        '''p = 0
        print(xi.shape)
        if xi.shape[2]==200:
            p = 1
        elif xi.shape[2]==100:
            p = 2
        elif xi.shape[2]==50:
            p = 3
        elif xi.shape[2]==25:
            p = 4
            
        if xi.shape[1] == 51:
            p = 1
        if xi.shape[1] == 51 and xi.shape[2] == 50:
            p = 2'''
        
               
        #print(xi.shape[2])
        #print(self.stride)
        #print(self.kerner_size)
        #p = max(0, math.ceil((xi.shape[2] - (xi.shape[2] - 1) * self.stride - self.kerner_size) / 2))
        
        xi = self.binfunc(xi)
        
        new_weight = self.binarization(self.weights3, self.scale)
        if ks == 5:
            new_weight = self.binarization(self.weights5, self.scale)
        elif ks == 7:
            new_weight = self.binarization(self.weights7, self.scale)
        elif ks == 9:
            new_weight = self.binarization(self.weights9, self.scale)
        
        conv2 = F.conv2d(xi, new_weight, self.bias, self.stride,
                        1 + (int)((ks-3)/2), self.dilation, self.groups)
        '''conv2 = F.conv2d(xi, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        # 输入张量大小
        input_size = conv2.shape

        # 目标输出大小
        desired_output_size = conv1.shape

        # 计算填充值
        padding_height = desired_output_size[-2] - input_size[-2]
        padding_width = desired_output_size[-1] - input_size[-1]

        # 使用填充操作扩展张量
        conv2 = F.pad(conv2, (0, padding_width, 0, padding_height))'''
        
        #print(conv1.shape)
        #print(conv2.shape)
        return conv1 + conv2
    
    @property
    def pre_kernel(self):
        return self._pre_kernel

class LBPConv3(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(LBPConv3, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')
        global layer
        self.binfunc = BinaryActivation()
        self.binarization = Binarization.apply
        self.generate_scale()
        
        layer = layer + 1
        ks = 3
        '''if layer <= 2:
            ks = 3
        elif layer <= 4 and layer >= 3:
            ks = 5
        elif layer <= 6:
            ks = 7
        elif layer <= 8:
            ks = 9'''
        
        if layer == 3:
            ks = 5
        elif layer == 5:
            ks = 7
        elif layer == 7:
            ks = 9
        
        if layer >= 8:
            layer = layer - 8
        
        self.lbp = 0.2
        self.lbp_channels = math.floor(in_channels * self.lbp)
        self.conv_channels = in_channels - self.lbp_channels
        #print(conv_channels)
        if self.lbp_channels > 0:
            #self.lbpconv = LBPConv(self.lbp_channels, planes, stride=stride, R=R)
            self.lbpconv = LBPConv2(in_channels=self.lbp_channels, out_channels=self.out_channels, kernel_size=ks, stride=stride, padding = 1 + (int)((ks-3)/2) )
            self.lbp_bn = nn.BatchNorm2d(out_channels)
        
        
        #print(layer)
        print('lbp channels: %d, conv channels: %d' % (self.lbp_channels, self.conv_channels))
        print('rprelu here')
        self.relu = RPReLU(out_channels)
        self.movelbp = LearnableBias(self.lbp_channels)
        self.move0 = LearnableBias(self.conv_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        #self.pre_conv = BiConv(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=1, stride=stride, padding=0)
        self.conv = BiConv(in_channels=self.conv_channels, out_channels=self.out_channels, kernel_size=ks, stride=stride, padding=1 + (int)((ks-3)/2) )
        #self.suf_conv = BiConv(in_channels=self.conv_channels, out_channels=self.out_channels, kernel_size=1, stride=stride, padding=0)
        #self.conv = BiConv(in_channels=self.conv_channels, out_channels=self.out_channels, kernel_size=3, stride=stride, padding=1)
    
    def generate_scale(self):
        self.scale = Parameter(torch.randn(self.out_channels, 1, 1, 1))
    
    def forward(self, x):
        #print(x.shape)
        #print(self.pre_kernel.shape)
        '''global layer
        if layer % 2 == 0: #对应部分还原卷积
            x = self.binfunc(x)

            new_weight = self.binarization(self.weight, self.scale)

            return F.conv2d(x, new_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)'''
        out_fuse = []
        #if self.ok == 0:
        if self.lbp_channels > 0:
            out = self.movelbp(x[:, :self.lbp_channels, :, :])
            out = self.lbpconv(out)
            out = self.lbp_bn(out)
            out_fuse.append(out)
            #out_fuse.append(self.lbp_bn(self.lbpconv(self.movelbp(x[:, :self.lbp_channels, :, :]))))
        if self.conv_channels > 0:
            out = self.move0(x[:, self.lbp_channels:, :, :])
            #out = self.pre_conv(out)
            out = self.conv(out)
            #out = self.suf_conv(out)
            out = self.bn1(out)
            out_fuse.append(out)
        if len(out_fuse) > 1:
            #print(out_fuse[0].shape)
            #print(out_fuse[1].shape)
            out = out_fuse[0] + out_fuse[1]
        else:
            out = out_fuse[0]                 
            #print(len(out_fuse))
        
        return out
    
class JLBPConv(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(JLBPConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')
        
        self.lbp = 0.2
        self.lbp_channels = math.floor(in_channels * self.lbp)
        self.conv_channels = in_channels - self.lbp_channels
        #print(conv_channels)
        if self.lbp_channels > 0:
            #self.lbpconv = LBPConv(self.lbp_channels, planes, stride=stride, R=R)
            #self.lbpconv = LBPConv(in_channels=self.lbp_channels, out_channels=self.out_channels, kernel_size=3, stride=stride)
            self.lbpconv = JLBPwBiConv(in_channels=self.lbp_channels, out_channels=self.out_channels, kernel_size=5, stride=stride)
            self.lbp_bn = nn.BatchNorm2d(out_channels)
        print('lbp channels: %d, conv channels: %d' % (self.lbp_channels, self.conv_channels))
        print('rprelu here')
        self.relu = RPReLU(out_channels)
        self.movelbp = LearnableBias(self.lbp_channels)
        self.move0 = LearnableBias(self.conv_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv = BiConv(in_channels=self.conv_channels, out_channels=self.out_channels, kernel_size=3, stride=stride, padding=1)
    
    def forward(self, x):
        #print(x.shape)
        #print(self.pre_kernel.shape)
        out_fuse = []
        #if self.ok == 0:
        if self.lbp_channels > 0:
            out = self.movelbp(x[:, :self.lbp_channels, :, :])
            out = self.lbpconv(x[:, :self.lbp_channels, :, :])
            out = self.lbp_bn(out)
            out_fuse.append(out)
            #out_fuse.append(self.lbp_bn(self.lbpconv(self.movelbp(x[:, :self.lbp_channels, :, :]))))
        if self.conv_channels > 0:
            out = self.move0(x[:, self.lbp_channels:, :, :])
            out = self.conv(out)
            out = self.bn1(out)
            out_fuse.append(out)
        if len(out_fuse) > 1:
            #print(out_fuse[0].shape)
            #print(out_fuse[1].shape)
            out = out_fuse[0] + out_fuse[1]
        else:
            out = out_fuse[0]                 
            #print(len(out_fuse))
        
        return out
    
class NILBPConv(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(NILBPConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')

        self.generate_scale()
        self.register_buffer('_pre_kernel', self.get_kernel()) # 8*C_in, 1, 3, 3
        self.binarization = Binarization.apply
        self.binfunc = BinaryActivation()
        self.out_channels = out_channels
        self.shape = (out_channels, 8 * in_channels // 1, 1, 1)
        self.weights = nn.Parameter(torch.Tensor(*self.shape))
        
    def generate_scale(self):
        self.scale = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def get_kernel(self):
        #NILBP
        kernel = torch.tensor( [[ 1-0.125,  -0.125,  -0.125,  -0.125,  0.,  -0.125,  -0.125,  -0.125,  -0.125],
                                [ -0.125,  1-0.125,  -0.125,  -0.125,  0.,  -0.125,  -0.125,  -0.125,  -0.125],
                                [ -0.125,  -0.125,  1-0.125,  -0.125,  0.,  -0.125,  -0.125,  -0.125,  -0.125],
                                [ -0.125,  -0.125,  -0.125,  1-0.125,  0.,  -0.125,  -0.125,  -0.125,  -0.125],
                                [ -0.125,  -0.125,  -0.125,  -0.125,  0.,  1-0.125,  -0.125,  -0.125,  -0.125],
                                [ -0.125,  -0.125,  -0.125,  -0.125,  0.,  -0.125,  1-0.125,  -0.125,  -0.125],
                                [ -0.125,  -0.125,  -0.125,  -0.125,  0.,  -0.125,  -0.125,  1-0.125,  -0.125],
                                [ -0.125,  -0.125,  -0.125,  -0.125,  0.,  -0.125,  -0.125,  -0.125,  1-0.125]])
        kernel = kernel.view(8, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        return kernel.contiguous()
    
    def forward(self, x):
        #print(x.shape)
        #print(self.pre_kernel.shape)
        x = F.conv2d(x, self.pre_kernel, stride=1, padding=1, 
                groups=self.in_channels, dilation=1) # b, 8*C_in, H, W
        #x = self.sign_layer(x)
        mask = torch.ones(x.size()[2:], device=x.device) * 1e-8
        mask[::2, :] *= -1
        mask[:, ::2] *= -1
        x = self.binfunc(x + mask.expand_as(x))

        new_weight = self.binarization(self.weights, self.scale)

        #return F.conv2d(x, new_weight, self.bias, self.stride,
        #                self.padding, self.dilation, self.groups)
        return F.conv2d(x, new_weight, stride=1, padding=0,
                groups=1)
    
    @property
    def pre_kernel(self):
        return self._pre_kernel

class CILBPConv(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CILBPConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')

        self.generate_scale()
        self.register_buffer('_pre_kernel', self.get_kernel()) # 8*C_in, 1, 3, 3
        self.binarization = Binarization.apply
        self.binfunc = BinaryActivation()
        self.out_channels = out_channels
        self.shape = (out_channels, 8 * in_channels // 1, 1, 1)
        self.weights = nn.Parameter(torch.Tensor(*self.shape))
        
    def generate_scale(self):
        self.scale = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def get_kernel(self):
        kernel = torch.tensor( [[ -1/9,  -1/9,  -1/9,  -1/9,  8/9.,  -1/9,  -1/9,  -1/9,  -1/9],
                        [ -1/9,  -1/9,  -1/9,  -1/9,  8/9.,  -1/9,  -1/9,  -1/9,  -1/9],
                        [ -1/9,  -1/9,  -1/9,  -1/9,  8/9.,  -1/9,  -1/9,  -1/9,  -1/9],
                        [ -1/9,  -1/9,  -1/9,  -1/9,  8/9.,  -1/9,  -1/9,  -1/9,  -1/9],
                        [ -1/9,  -1/9,  -1/9,  -1/9,  8/9.,  -1/9,  -1/9,  -1/9,  -1/9],
                        [ -1/9,  -1/9,  -1/9,  -1/9,  8/9.,  -1/9,  -1/9,  -1/9,  -1/9],
                        [ -1/9,  -1/9,  -1/9,  -1/9,  8/9.,  -1/9,  -1/9,  -1/9,  -1/9],
                        [ -1/9,  -1/9,  -1/9,  -1/9,  8/9.,  -1/9,  -1/9,  -1/9,  -1/9]])
        kernel = kernel.view(8, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        return kernel.contiguous()
    
    def forward(self, x):
        #print(x.shape)
        #print(self.pre_kernel.shape)
        x = F.conv2d(x, self.pre_kernel, stride=1, padding=1, 
                groups=self.in_channels, dilation=1) # b, 8*C_in, H, W
        #x = self.sign_layer(x)
        mask = torch.ones(x.size()[2:], device=x.device) * 1e-8
        mask[::2, :] *= -1
        mask[:, ::2] *= -1
        x = self.binfunc(x + mask.expand_as(x))

        new_weight = self.binarization(self.weights, self.scale)

        #return F.conv2d(x, new_weight, self.bias, self.stride,
        #                self.padding, self.dilation, self.groups)
        return F.conv2d(x, new_weight, stride=1, padding=0,
                groups=1)
    
    @property
    def pre_kernel(self):
        return self._pre_kernel

class CSLBPConv(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CSLBPConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')

        self.generate_scale()
        self.register_buffer('_pre_kernel', self.get_kernel()) # 8*C_in, 1, 3, 3
        self.binarization = Binarization.apply
        self.binfunc = BinaryActivation()
        self.out_channels = out_channels
        self.shape = (out_channels, 8 * in_channels // 1, 1, 1)
        self.weights = nn.Parameter(torch.Tensor(*self.shape))
        
    def generate_scale(self):
        self.scale = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def get_kernel(self):
        kernel = torch.tensor([[ 1.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  -1.],
                                [ 0.,  1.,  0.,  0.,  0.,  0., 0.,  -1.,  0.],
                                [ 0.,  0.,  1.,  0.,  0.,  0.,  -1., 0.,  0.],
                                [ 0.,  0.,  0.,  1.,  0.,  -1.,  0.,  0., 0.],
                                [ 0.,  0.,  0.,  -1.,  0.,  1.,  0.,  0.,  0.],
                                [ 0.,  0.,  -1.,  0.,  0.,  0.,  1.,  0.,  0.],
                                [ 0.,  -1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
                                [ -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
        kernel = kernel.view(8, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        return kernel.contiguous()
    
    def forward(self, x):
        #print(x.shape)
        #print(self.pre_kernel.shape)
        x = F.conv2d(x, self.pre_kernel, stride=1, padding=1, 
                groups=self.in_channels, dilation=1) # b, 8*C_in, H, W
        #x = self.sign_layer(x)
        mask = torch.ones(x.size()[2:], device=x.device) * 1e-8
        mask[::2, :] *= -1
        mask[:, ::2] *= -1
        x = self.binfunc(x + mask.expand_as(x))

        new_weight = self.binarization(self.weights, self.scale)

        #return F.conv2d(x, new_weight, self.bias, self.stride,
        #                self.padding, self.dilation, self.groups)
        return F.conv2d(x, new_weight, stride=1, padding=0,
                groups=1)
    
    @property
    def pre_kernel(self):
        return self._pre_kernel
    
class NICILBP(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(NICILBP, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')
        
        self.lbp = 0.2
        self.lbp_channels = math.floor(in_channels * self.lbp)
        self.conv_channels = in_channels - self.lbp_channels
        #print(conv_channels)
        if self.lbp_channels > 0:
            #self.lbpconv = LBPConv(self.lbp_channels, planes, stride=stride, R=R)
            self.nilbpconv = NILBPConv(in_channels=math.floor(self.lbp_channels*0.5), out_channels=self.out_channels, kernel_size=3, stride=stride)
            self.lbpconv = JLBPwBiConv(in_channels=self.lbp_channels-math.floor(self.lbp_channels*0.5), out_channels=self.out_channels, kernel_size=5, stride=stride)
            self.lbp_bn1 = nn.BatchNorm2d(out_channels)
            self.lbp_bn2 = nn.BatchNorm2d(out_channels)
        print('lbp channels: %d, conv channels: %d' % (self.lbp_channels, self.conv_channels))
        print('rprelu here')
        self.relu = RPReLU(out_channels)
        self.movelbp1 = LearnableBias(math.floor(self.lbp_channels*0.5))
        self.movelbp2 = LearnableBias(self.lbp_channels-math.floor(self.lbp_channels*0.5))
        self.move0 = LearnableBias(self.conv_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv = BiConv(in_channels=self.conv_channels, out_channels=self.out_channels, kernel_size=3, stride=stride, padding=1)
    
    def forward(self, x):
        #print(x.shape)
        #print(self.pre_kernel.shape)
        out_fuse = []
        #if self.ok == 0:
        if self.lbp_channels > 0:
            out = self.movelbp1(x[:, :math.floor(self.lbp_channels*0.5), :, :])
            out = self.nilbpconv(x[:, :math.floor(self.lbp_channels*0.5), :, :])
            out = self.lbp_bn1(out)
            out_fuse.append(out)
            #out_fuse.append(self.lbp_bn(self.lbpconv(self.movelbp(x[:, :self.lbp_channels, :, :]))))
        if self.lbp_channels > 0:
            out = self.movelbp2(x[:, math.floor(self.lbp_channels*0.5):self.lbp_channels, :, :])
            out = self.lbpconv(x[:, math.floor(self.lbp_channels*0.5):self.lbp_channels, :, :])
            out = self.lbp_bn2(out)
            out_fuse.append(out)
        if self.conv_channels > 0:
            out = self.move0(x[:, self.lbp_channels:, :, :])
            out = self.conv(out)
            out = self.bn1(out)
            out_fuse.append(out)
        if len(out_fuse) > 2:
            #print(out_fuse[0].shape)
            #print(out_fuse[1].shape)
            out = out_fuse[0] + out_fuse[1] + out_fuse[2]
        else:
            out = out_fuse[0]                 
            #print(len(out_fuse))
        
        return out

class MLBPConv3(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MLBPConv3, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')
        
        self.lbp = 0.2
        self.lbp_channels = math.floor(in_channels * self.lbp)
        self.conv_channels = in_channels - self.lbp_channels
        #print(conv_channels)
        self.kernel_size = kernel_size
        p = 0
        print(kernel_size)
        if kernel_size == (3, 3):
            p = 0
        elif kernel_size == (5, 5):
            p = 1
        elif kernel_size == (7, 7):
            p = 2
        elif kernel_size == (9, 9):
            p = 3

        if self.lbp_channels > 0:
            #self.lbpconv = LBPConv(self.lbp_channels, planes, stride=stride, R=R)
            self.lbpconv = LBPConv(in_channels=self.lbp_channels, out_channels=self.out_channels, kernel_size=kernel_size, stride=stride, padding=p)
            self.lbp_bn = nn.BatchNorm2d(out_channels)
        print('lbp channels: %d, conv channels: %d' % (self.lbp_channels, self.conv_channels))
        print('rprelu here')
        self.relu = RPReLU(out_channels)
        self.movelbp = LearnableBias(self.lbp_channels)
        self.move0 = LearnableBias(self.conv_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv = BiConv(in_channels=self.conv_channels, out_channels=self.out_channels, kernel_size=3, stride=stride, padding=1)
    
    def forward(self, x):
        #print(x.shape)
        #print(self.pre_kernel.shape)
        out_fuse = []
        #if self.ok == 0:
        if self.lbp_channels > 0:
            out = self.movelbp(x[:, :self.lbp_channels, :, :])
            out = self.lbpconv(x[:, :self.lbp_channels, :, :])
            out = self.lbp_bn(out)
            out_fuse.append(out)
            #out_fuse.append(self.lbp_bn(self.lbpconv(self.movelbp(x[:, :self.lbp_channels, :, :]))))
        if self.conv_channels > 0:
            out = self.move0(x[:, self.lbp_channels:, :, :])
            out = self.conv(out)
            out = self.bn1(out)
            out_fuse.append(out)
        if len(out_fuse) > 1:
            #print(self.kernel_size)
            #print(out_fuse[0].shape)
            #print(out_fuse[1].shape)
            out = out_fuse[0] + out_fuse[1]
        else:
            out = out_fuse[0]                 
            #print(len(out_fuse))
        
        return out

'''
class LBPConv3(_ConvNd):
    
    #Baee layer class for modulated convolution
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(LBPConv3, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')
        
        self.lbp = 0.2
        self.lbp_channels = math.floor(in_channels * self.lbp)
        self.conv_channels = in_channels - self.lbp_channels
        self.generate_scale()
        self.register_buffer('_pre_kernel', self.get_kernel()) # 8*C_in, 1, 3, 3
        self.binarization = Binarization.apply
        self.binfunc = BinaryActivation()
        self.out_channels = out_channels
        self.shape1 = (out_channels, 8 * self.lbp_channels // groups, 1, 1)
        self.shape2 = (out_channels, self.conv_channels // groups, 1, 1)
        self.weights1 = nn.Parameter(torch.Tensor(*self.shape1))
        self.weights2 = nn.Parameter(torch.Tensor(*self.shape2))
        
    def generate_scale(self):
        self.scale1 = Parameter(torch.randn(self.out_channels, 1, 1, 1))
        self.scale2 = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def get_kernel(self):
        kernel = torch.zeros(8, 9)
        indices = torch.stack([torch.tensor([0, 1, 2, 3, 5, 6, 7, 8]).long(),
            torch.ones(8).long() * 4]).transpose(1, 0)
        values = torch.ones(8,2)
        values[:, 1] = -1
        kernel.scatter_(1, indices, values)
        kernel = kernel.view(8, 1, 3, 3).repeat(self.lbp_channels, 1, 1, 1)
        return kernel.contiguous()
    
    def forward(self, x):
        #print(x.shape)
        #print(self.pre_kernel.shape)
        xi = x[:, self.lbp_channels:, :, :]
        x = x[:, :self.lbp_channels, :, :]
        #out_fuse = []
        x = F.conv2d(x, self.pre_kernel, stride=1, padding=1, 
                groups=self.lbp_channels, dilation=1) # b, 8*C_in, H, W
        #x = self.sign_layer(x)
        mask = torch.ones(x.size()[2:], device=x.device) * 1e-8
        mask[::2, :] *= -1
        mask[:, ::2] *= -1
        x = self.binfunc(x + mask.expand_as(x))

        new_weight1 = self.binarization(self.weights1, self.scale1)

        #return F.conv2d(x, new_weight, self.bias, self.stride,
        #                self.padding, self.dilation, self.groups)
        conv1 = F.conv2d(x, new_weight1, stride=1, padding=0,
                groups=1)
        
        xi = self.binfunc(xi)

        new_weight2 = self.binarization(self.weights2, self.scale2)
             
        conv2 = F.conv2d(xi, new_weight2, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
        return conv1 + conv2
    
    @property
    def pre_kernel(self):
        return self._pre_kernel'''

class BiConv_PDC(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BiConv_PDC, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')

        self.generate_scale()
        self.binarization = Binarization.apply
        self.binfunc = BinaryActivation()
        self.out_channels = out_channels

        self.conv = BiConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = 0.2
        
    def generate_scale(self):
        self.scale = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def forward(self, x):

        #x = self.binfunc(x)

        #new_weight = self.binarization(self.weight, self.scale)

        #conv_result = F.conv2d(x, new_weight, self.bias, self.stride,
        #                       self.padding, self.dilation, self.groups)
        
        out_normal = self.conv(x)
        '''center_i = self.kernel_size[0] // 2
        center_j = self.kernel_size[1] // 2
        center_pixel = new_weight[:, :, center_i, center_j]
        
        # Get the size of the central region for each output pixel
        central_region_size = conv_result.size()[2:]

        # Expand center_pixel to match the size of conv_result's central region
        center_pixel_expanded = center_pixel.view(-1, self.out_channels, 1, 1)
        center_pixel_expanded = center_pixel_expanded.expand(-1, -1, *central_region_size)

        # Repeat center_pixel values to match the batch size
        center_pixel_expanded = center_pixel_expanded.expand(x.size(0), -1, -1, -1)

        # Subtract center_pixel values from the central region of conv_result
        modified_output = conv_result - center_pixel_expanded'''
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        xx = self.binfunc(x)
        out_diff = F.conv2d(input=xx, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

        #return modified_output
        return out_normal - self.theta * out_diff

class BiConv_PDC_multi(_ConvNd):
    '''
    Base layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        k2 = _pair(kernel_size + 2)
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        p2 = _pair(padding + 1)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BiConv_PDC_multi, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')

        self.generate_scale()
        self.binarization = Binarization.apply
        self.binfunc = BinaryActivation()
        self.out_channels = out_channels

        self.conv = BiConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = 0.2
        
        # 添加多尺度卷积
        self.conv_multiscale = BiConv(in_channels, out_channels, kernel_size=k2, stride=stride, padding=p2, dilation=dilation, groups=groups, bias=bias)
        
        # 添加池化层
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def generate_scale(self):
        self.scale = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def forward(self, x):
        
        # 多尺度卷积操作
        conv_multiscale_output = self.conv_multiscale(x)
        
        #x = self.binfunc(x)

        #new_weight = self.binarization(self.weight, self.scale)

        #conv_result = F.conv2d(x, new_weight, self.bias, self.stride,
        #                       self.padding, self.dilation, self.groups)

        out_normal = self.conv(x)
        
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        xx = self.binfunc(x)
        out_diff = F.conv2d(input=xx, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

        return out_normal - self.theta * out_diff + conv_multiscale_output
        #+ pooled_output

class Sign(nn.Module):

    def __init__(self, bound=1.2, binary=True):
        super(Sign, self).__init__()
        self.bound = bound
        self.binary = binary

    def reset_state(self, bound, binary):
        self.bound = bound
        self.binary = binary

    def forward(self, x):
        if not self.binary:
            return x
        if not self.training:
            return torch.sign(x)

        #print('activation before sign: ', x, flush=True)
        out = torch.clamp(x, -self.bound, self.bound)
        out_forward = torch.sign(x)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        y = out_forward.detach() + out - out.detach()
        return y

class BiConv_LBP(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.kernel_size = 3
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = self.dilation
        self.bound = 1.0
        super(BiConv_LBP, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode='zeros')

        self.generate_scale()
        self.binarization = Binarization.apply
        self.binfunc = BinaryActivation()
        self.out_channels = out_channels
        self.sign_layer = Sign()
        self.shape = (out_channels, 8 * in_channels // groups, 1, 1)
        self.weights = nn.Parameter(torch.Tensor(*self.shape))
        self.bound = 1.0

        self.register_buffer('_pre_kernel', self.get_kernel()) # 8*C_in, 1, 3, 3
    
    def get_kernel(self):
        kernel = torch.zeros(8, 9)
        indices = torch.stack([torch.tensor([0, 1, 2, 3, 5, 6, 7, 8]).long(),
            torch.ones(8).long() * 4]).transpose(1, 0)
        values = torch.ones(8,2)
        values[:, 1] = -1
        kernel.scatter_(1, indices, values)
        kernel = kernel.view(8, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        return kernel.contiguous()

    def reset_state(self, bound):
        self.bound = bound

    def generate_scale(self):
        self.scale = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def forward(self, x):
        x = F.conv2d(x, self.pre_kernel, stride=self.stride, padding=self.padding, 
                groups=self.in_channels, dilation=self.dilation) # b, 8*C_in, H, W
        #x = self.sign_layer(x)
        mask = torch.ones(x.size()[2:], device=x.device) * 1e-8
        mask[::2, :] *= -1
        mask[:, ::2] *= -1
        x = self.sign_layer(x + mask.expand_as(x))
        #assert torch.sum(x==0) < 1, '0 appears in binary activations, got %d 0s' % torch.sum(x==0).item()

        clipped_weights = torch.clamp(self.weights, -self.bound, self.bound)

        binary_weights_no_grad = torch.sign(self.weights).detach()
        binary_weights = binary_weights_no_grad + \
                clipped_weights - clipped_weights.detach()

        x = F.conv2d(x, binary_weights, stride=1, padding=0,
                groups=self.groups) # b, C_out, H, W
        return x

    @property
    def pre_kernel(self):
        return self._pre_kernel

class Conv2d_cd(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_cd, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups=groups, bias=False, padding_mode='zeros')
        #super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = 0.7

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class BiCDC(_ConvNd): #15.2
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BiCDC, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups=groups, bias=False, padding_mode='zeros')
        #super(Conv2d_cd, self).__init__()
        self.conv = BiConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = 0.7
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff
            #return out_normal - torch.sigmoid(out_normal) * out_diff

class BiPDC(_ConvNd): #15.2
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', bound = 1.2):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BiPDC, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups=groups, bias=False, padding_mode='zeros')
        #super(Conv2d_cd, self).__init__()
        self.conv = BiConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = 0.2
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.sign_layer = Sign()
        self.bound = bound

        self.register_buffer('_pre_kernel', self.get_kernel()) # 8*C_in, 1, 3, 3
        self.shape = (out_channels, 8 * in_channels // groups, 1, 1)
        self.weights = nn.Parameter(torch.Tensor(*self.shape))
        torch.nn.init.xavier_normal_(self.weights, gain=2.0)

    def get_kernel(self):
        kernel = torch.zeros(8, 9)
        indices = torch.stack([torch.tensor([0, 1, 2, 3, 5, 6, 7, 8]).long(),
            torch.ones(8).long() * 4]).transpose(1, 0)
        values = torch.ones(8,2)
        values[:, 1] = -1
        kernel.scatter_(1, indices, values)
        kernel = kernel.view(8, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        return kernel.contiguous()

    def forward(self, x):
        x = F.conv2d(x, self.pre_kernel, stride=self.stride, padding=self.padding, 
                groups=self.in_channels, dilation=self.dilation)  # 第一次卷积操作
        #x = self.sign_layer(x)
        mask = torch.ones(x.size()[2:], device=x.device) * 1e-8
        mask[::2, :] *= -1
        mask[:, ::2] *= -1
        x = self.sign_layer(x + mask.expand_as(x))  # 添加二值化时的掩码
        #assert torch.sum(x==0) < 1, '0 appears in binary activations, got %d 0s' % torch.sum(x==0).item()

        clipped_weights = torch.clamp(self.weights, -self.bound, self.bound)  # 将权重限制在边界内

        binary_weights_no_grad = torch.sign(self.weights).detach()  # 使用sign函数生成二值权重（不带梯度）
        binary_weights = binary_weights_no_grad + \
                clipped_weights - clipped_weights.detach()  # 生成带梯度的二值权重

        x = F.conv2d(x, binary_weights, stride=1, padding=0,
                groups=self.groups)  # 使用二值权重进行卷积操作

        return x
        
        out_normal = self.conv(x)
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
        return x - 0.2 * out_diff
        
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff
            #return out_normal - torch.sigmoid(out_normal) * out_diff
    
    @property
    def pre_kernel(self):
        return self._pre_kernel
    
class BiPDCv2(_ConvNd): #15.2
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', bound = 1.2):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BiPDCv2, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups=groups, bias=False, padding_mode='zeros')
        #super(Conv2d_cd, self).__init__()
        self.conv = BiConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.sign_layer = Sign()
        self.bound = bound
        self.theta = 0.2

        self.register_buffer('_pre_kernel', self.get_kernel(in_channels))
        self.shape = (out_channels, 8 * in_channels // groups, 1, 1)
        self.weights = nn.Parameter(torch.Tensor(*self.shape))
        nn.init.xavier_normal_(self.weights, gain=2.0)

    def get_kernel(self):
        kernel = torch.zeros(8, 9)
        indices = torch.stack([torch.tensor([0, 1, 2, 3, 5, 6, 7, 8]).long(),
            torch.ones(8).long() * 4]).transpose(1, 0)
        values = torch.ones(8,2)
        values[:, 1] = -1
        kernel.scatter_(1, indices, values)
        kernel = kernel.view(8, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        return kernel.contiguous()

    def forward(self, x):
        x = F.conv2d(x, self.pre_kernel, stride=self.stride, padding=self.padding, 
                groups=self.in_channels, dilation=self.dilation)  # 第一次卷积操作
        #x = self.sign_layer(x)
        mask = torch.ones(x.size()[2:], device=x.device) * 1e-8
        mask[::2, :] *= -1
        mask[:, ::2] *= -1
        x = self.sign_layer(x + mask.expand_as(x))  # 添加二值化时的掩码
        #assert torch.sum(x==0) < 1, '0 appears in binary activations, got %d 0s' % torch.sum(x==0).item()

        clipped_weights = torch.clamp(self.weights, -self.bound, self.bound)  # 将权重限制在边界内

        binary_weights_no_grad = torch.sign(self.weights).detach()  # 使用sign函数生成二值权重（不带梯度）
        binary_weights = binary_weights_no_grad + \
                clipped_weights - clipped_weights.detach()  # 生成带梯度的二值权重

        x = self.conv(x)  # 使用二值权重进行卷积操作
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
        return x - self.theta * out_diff
    
    @property
    def pre_kernel(self):
        return self._pre_kernel

class BiConv2d_cd(_ConvNd): #15.9
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BiConv2d_cd, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups=groups, bias=False, padding_mode='zeros')
        #super(Conv2d_cd, self).__init__()
        self.conv = BiConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = 0.8
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()

    def forward(self, x):
        out_normal = self.conv(x)
        #特征激励机制
        m1 = self.sigmoid(x) + 0.5 # [0.5, 1.5]
        m2 = self.elu(x)
        reward = 2 * (m1 - 1) + 1
        yi = torch.zeros_like(m1)
        yi = yi + 1
        punishment = torch.where(m1 > 1 , m1 , yi)
        m1 = torch.where(m1 > 1 , reward , punishment) # 低特征值保持不变，高特征值持续激励
        x = torch.multiply(m1, m2)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff
            #return out_normal - torch.sigmoid(out_normal) * out_diff

class BiConv2d_pool(_ConvNd): #15.9
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BiConv2d_pool, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups=groups, bias=False, padding_mode='zeros')
        #super(Conv2d_cd, self).__init__()
        self.conv = BiConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = 0.7
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.pool = F.avg_pool2d

    def forward(self, x):
        out_diff = self.pool(x, kernel_size=3, stride=self.stride, padding = 1) - x/9
        x = x + out_diff
        out_normal = self.conv(x)
        #特征激励机制
        m1 = self.sigmoid(x) + 0.5 # [0.5, 1.5]
        m2 = self.elu(x)
        reward = 2 * (m1 - 1) + 1
        yi = torch.zeros_like(m1)
        yi = yi + 1
        punishment = torch.where(m1 > 1 , m1 , yi)
        m1 = torch.where(m1 > 1 , reward , punishment) # 低特征值保持不变，高特征值持续激励
        x = torch.multiply(m1, m2)
        
        #print(x.shape)
        #print(out_normal.shape)
        #print(out_diff.shape)
        #out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

        return out_normal
        #return out_normal - torch.sigmoid(out_normal) * out_diff

class BiConv2d_cs(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BiConv2d_cs, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups=groups, bias=False, padding_mode='zeros')
        #super(Conv2d_cd, self).__init__()
        self.conv = BiConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = 0.7
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()

    def forward(self, x):
        out_normal = self.conv(x)
        #特征激励机制
        '''m1 = self.sigmoid(x) + 0.5 # [0.5, 1.5]
        m2 = self.elu(x)
        reward = 2 * (m1 - 1) + 1
        yi = torch.zeros_like(m1)
        yi = yi + 1
        punishment = torch.where(m1 > 1 , m1 , yi)
        m1 = torch.where(m1 > 1 , reward , punishment) # 低特征值保持不变，高特征值持续激励
        x = torch.multiply(m1, m2)'''

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            conv_weight1 = self.conv.weight[:,:,0,0]
            conv_weight1 = conv_weight1[:,:,None,None]
            conv_weight2 = self.conv.weight[:,:,0,1]
            conv_weight2 = conv_weight2[:,:,None,None]
            conv_weight3 = self.conv.weight[:,:,0,2]
            conv_weight3 = conv_weight3[:,:,None,None]
            conv_weight4 = self.conv.weight[:,:,1,0]
            conv_weight4 = conv_weight4[:,:,None,None]
            conv_weight5 = self.conv.weight[:,:,1,1]
            conv_weight5 = conv_weight5[:,:,None,None]
            conv_weight6 = self.conv.weight[:,:,1,2]
            conv_weight6 = conv_weight6[:,:,None,None]
            conv_weight7 = self.conv.weight[:,:,2,0]
            conv_weight7 = conv_weight7[:,:,None,None]
            conv_weight8 = self.conv.weight[:,:,2,1]
            conv_weight8 = conv_weight8[:,:,None,None]
            conv_weight9 = self.conv.weight[:,:,2,2]
            conv_weight9 = conv_weight9[:,:,None,None]

            conv_weight = conv_weight1 + conv_weight2 + conv_weight3 + conv_weight4 - conv_weight6 - conv_weight7 - conv_weight8 - conv_weight9
            out_diff = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            return out_normal - self.theta * out_diff
            #return out_normal - torch.sigmoid(out_normal) * out_diff

class BiConv2d_csenc(_ConvNd): #16.6
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BiConv2d_csenc, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups=groups, bias=False, padding_mode='zeros')
        #super(Conv2d_cd, self).__init__()
        self.conv = BiConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = 0.7
        self.sigmoid = nn.Tanh()
        self.elu = nn.ReLU()
        #self.convcs = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        out_normal = self.conv(x)
        #特征激励机制
        m1 = self.sigmoid(x) + 1 # [1, 2]
        #m1 = self.sigmoid(x) + 0.5 # [0.5, 1.5]
        m2 = self.elu(x)
        reward = m1
        #reward = 2 * (m1 - 1) + 1
        yi = torch.zeros_like(m1)
        yi = yi + 1
        punishment = torch.where(m1 > 1 , m1 , yi)
        m1 = torch.where(m1 > 1 , reward , punishment) # 低特征值保持不变，高特征值持续激励
        x = torch.multiply(m1, m2)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
            #print(self.conv.weight[:,:,:,0].shape)
            '''conv_weight1 = torch.cat((self.conv.weight[:,:,0,0], tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros), 2)
            #conv_weight1 = conv_weight1.sum(2)
            conv_weight2 = torch.cat((tensor_zeros, self.conv.weight[:,:,0,1], tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros), 2)
            #conv_weight2 = conv_weight2.sum(2)
            conv_weight3 = torch.cat((tensor_zeros, tensor_zeros, self.conv.weight[:,:,0,2], tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros), 2)
            #conv_weight3 = conv_weight3.sum(2)
            conv_weight4 = torch.cat((tensor_zeros, tensor_zeros, tensor_zeros, self.conv.weight[:,:,1,0], tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros), 2)
            #conv_weight4 = conv_weight4.sum(2)
            conv_weight5 = torch.cat((tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, self.conv.weight[:,:,1,1], tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros), 2)
            #conv_weight5 = conv_weight5.sum(2)
            conv_weight6 = torch.cat((tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, self.conv.weight[:,:,1,2], tensor_zeros, tensor_zeros, tensor_zeros), 2)
            #conv_weight6 = conv_weight6.sum(2)
            conv_weight7 = torch.cat((tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, self.conv.weight[:,:,2,0], tensor_zeros, tensor_zeros), 2)
            #conv_weight7 = conv_weight7.sum(2)
            conv_weight8 = torch.cat((tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, self.conv.weight[:,:,2,1], tensor_zeros), 2)
            #conv_weight8 = conv_weight8.sum(2)
            conv_weight9 = torch.cat((tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, tensor_zeros, self.conv.weight[:,:,2,2]), 2)
            #conv_weight9 = conv_weight9.sum(2)'''

            conv_weight1 = self.conv.weight[:,:,0,0]
            #conv_weight1 = conv_weight1.sum(2)
            conv_weight1 = conv_weight1[:,:,None,None]
            conv_weight2 = self.conv.weight[:,:,0,1]
            #conv_weight2 = conv_weight2.sum(2)
            conv_weight2 = conv_weight2[:,:,None,None]
            conv_weight3 = self.conv.weight[:,:,0,2]
            #conv_weight3 = conv_weight3.sum(2)
            conv_weight3 = conv_weight3[:,:,None,None]
            conv_weight4 = self.conv.weight[:,:,1,0]
            conv_weight4 = conv_weight4[:,:,None,None]
            #conv_weight4 = conv_weight4.sum(2)
            conv_weight5 = self.conv.weight[:,:,1,1]
            conv_weight5 = conv_weight5[:,:,None,None]
            #conv_weight5 = conv_weight5.sum(2)
            conv_weight6 = self.conv.weight[:,:,1,2]
            conv_weight6 = conv_weight6[:,:,None,None]
            #conv_weight6 = conv_weight6.sum(2)
            conv_weight7 = self.conv.weight[:,:,2,0]
            conv_weight7 = conv_weight7[:,:,None,None]
            #conv_weight7 = conv_weight7.sum(2)
            conv_weight8 = self.conv.weight[:,:,2,1]
            conv_weight8 = conv_weight8[:,:,None,None]
            #conv_weight8 = conv_weight8.sum(2)
            conv_weight9 = self.conv.weight[:,:,2,2]
            conv_weight9 = conv_weight9[:,:,None,None]
            #conv_weight9 = conv_weight9.sum(2)
            conv_weight = conv_weight1 + conv_weight2 + conv_weight3 + conv_weight4 - conv_weight6 - conv_weight7 - conv_weight8 - conv_weight9
            #kernel_diff = self.conv.weight.sum(2).sum(2)
            #kernel_diff = kernel_diff[:, :, None, None]
            #out_diff1 = F.conv2d(input=x, weight=conv_weight9, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            #out_diff2 = F.conv2d(input=x, weight=conv_weight8, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            #out_diff3 = F.conv2d(input=x, weight=conv_weight7, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            #out_diff4 = F.conv2d(input=x, weight=conv_weight6, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            #out_diff1 = F.conv2d(input=x, weight=conv_weight1, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            #out_diff2 = F.conv2d(input=x, weight=conv_weight2, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            #out_diff3 = F.conv2d(input=x, weight=conv_weight3, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            #out_diff4 = F.conv2d(input=x, weight=conv_weight4, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            #out_diff5 = F.conv2d(input=x, weight=conv_weight5, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            #out_diff6 = F.conv2d(input=x, weight=conv_weight6, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            #out_diff7 = F.conv2d(input=x, weight=conv_weight7, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            #out_diff8 = F.conv2d(input=x, weight=conv_weight8, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            #out_diff9 = F.conv2d(input=x, weight=conv_weight9, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            #y = self.convcs(x)
            out_diff = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            #out_diff = out_diff1 + out_diff2 + out_diff3 + out_diff4 - out_diff6 - out_diff7 - out_diff8 - out_diff9

            #print(out_normal.shape)
            #print(out_diff.shape)


            return out_normal - self.theta * out_diff
            #return out_normal - torch.sigmoid(out_normal) * out_diff
'''
class LBPConv(nn.Module):
    
    def __init__(self, in_chn, out_chn, stride=1, 
            groups=1, dilation=1, bound=1.2, R=None):
        super(LBPConv, self).__init__()
        self.in_channels = in_chn
        self.out_channels = out_chn
        self.kernel_size = 3
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = self.dilation
        self.bound = bound
        self.R = R
        self.sign_layer = Sign()

        self.register_buffer('_pre_kernel', self.get_kernel()) # 8*C_in, 1, 3, 3
        self.shape = (out_chn, 8 * in_chn // groups, 1, 1)
        self.weights = nn.Parameter(torch.Tensor(*self.shape))
        torch.nn.init.xavier_normal_(self.weights, gain=2.0)

    def get_kernel(self):
        kernel = torch.zeros(8, 9)
        indices = torch.stack([torch.tensor([0, 1, 2, 3, 5, 6, 7, 8]).long(),
            torch.ones(8).long() * 4]).transpose(1, 0)
        values = torch.ones(8,2)
        values[:, 1] = -1
        kernel.scatter_(1, indices, values)
        kernel = kernel.view(8, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        return kernel.contiguous()

    def reset_state(self, bound):
        self.bound = bound

    def forward(self, x):
        x = F.conv2d(x, self.pre_kernel, stride=self.stride, padding=self.padding, 
                groups=self.in_channels, dilation=self.dilation) # b, 8*C_in, H, W
        #x = self.sign_layer(x)
        mask = torch.ones(x.size()[2:], device=x.device) * 1e-8
        mask[::2, :] *= -1
        mask[:, ::2] *= -1
        x = self.sign_layer(x + mask.expand_as(x))
        #assert torch.sum(x==0) < 1, '0 appears in binary activations, got %d 0s' % torch.sum(x==0).item()

        clipped_weights = torch.clamp(self.weights, -self.bound, self.bound)

        binary_weights_no_grad = torch.sign(self.weights).detach()
        binary_weights = binary_weights_no_grad + \
                clipped_weights - clipped_weights.detach()

        x = F.conv2d(x, binary_weights, stride=1, padding=0,
                groups=self.groups) # b, C_out, H, W
        return x
        
    @property
    def pre_kernel(self):
        return self._pre_kernel
'''
class RPReLU(nn.Module):
    """
        ReAct PReLU function.
    """
    def __init__(self, chn, init=0.25):
        super(RPReLU, self).__init__()
        self.init = init
        self.gamma = nn.Parameter(torch.ones(1, chn, 1, 1) * init)
        self.beta1 = nn.Parameter(torch.zeros(1, chn, 1, 1))
        self.beta2 = nn.Parameter(torch.zeros(1, chn, 1, 1))

    def forward(self, x):

        x = x + self.beta1.expand_as(x)
        x = torch.where(x > 0, x + self.beta2.expand_as(x), x * self.gamma + self.beta2.expand_as(x))

        return x
    
class BConv(nn.Module):
    """
        convolution with binary weights and binary activations
    """
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, 
            groups=1, dilation=1, bound=1.2, binary=True, R=None, sign_layer=None):
        super(BConv, self).__init__()
        self.in_channels = in_chn
        self.out_channels = out_chn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.bound = bound
        self.binary = binary
        self.R = R
        if sign_layer is None:
            self.sign_layer = Sign()
        else:
            self.sign_layer = sign_layer(in_chn)

        self.shape = (out_chn, in_chn // groups, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.Tensor(*self.shape))
        torch.nn.init.xavier_normal_(self.weights, gain=2.0)

    def reset_state(self, bound, binary):
        self.bound = bound
        self.binary = binary
        self.sign_layer.reset_state(bound, binary)

    def forward(self, x):

        if not self.binary:
            return F.conv2d(x, self.weights, stride=self.stride, 
                    padding=self.padding, groups=self.groups, dilation=self.dilation)

        #print('weight before sign: ', self.weights, flush=True)
        x = self.sign_layer(x)
        #assert torch.sum(x==0) < 1, '0 appears in binary activations, got %d 0s' % torch.sum(x==0).item()

        clipped_weights = torch.clamp(self.weights, -self.bound, self.bound)

        binary_weights_no_grad = torch.sign(self.weights).detach()

        binary_weights = binary_weights_no_grad + \
                clipped_weights - clipped_weights.detach()
        #print('activation: ', x, flush=True)
        #print('weight: ', binary_weights, flush=True)
        out = F.conv2d(x, binary_weights, stride=self.stride, 
                padding=self.padding, groups=self.groups, dilation=self.dilation)
        return out

class LBPConvLayer(nn.Module):
    """
    结合自定义卷积和激活函数的卷积层。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', bound = 1.2,
                 lbp_ratio=0.0, R=None):
        super(LBPConvLayer, self).__init__()

        assert lbp_ratio >= 0.0 and lbp_ratio <= 1.0, 'lbp_ratio应该在 [0, 1] 范围内'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 3
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = self.dilation
        self.bound = bound
        self.R = R

        self.lbp_ratio = lbp_ratio
        self.conv_ratio = 1.0 - lbp_ratio

        lbp_channels = int(in_channels * lbp_ratio)
        self.lbp_channels = lbp_channels
        conv_channels = in_channels - lbp_channels

        if self.lbp_ratio > 0:
            self.lbpconv = LBPConv(lbp_channels, out_channels, kernel_size, stride=stride, R=R)
            self.lbp_bn = nn.BatchNorm2d(out_channels)
        if self.conv_ratio > 0:
            self.bconv = BConv(conv_channels, out_channels, kernel_size, stride=stride, R=R)
            self.bn = nn.BatchNorm2d(out_channels)
        
        self.rprelu = RPReLU(out_channels)

        self.shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.Tensor(*self.shape))
        torch.nn.init.xavier_normal_(self.weights, gain=2.0)

    def forward(self, x):
        out_fuse = []
        if self.lbp_ratio > 0:
            out_fuse.append(self.lbp_bn(self.lbpconv(x[:, :self.lbp_channels, :, :])))
        if self.conv_ratio > 0:
            out_fuse.append(self.bn(self.bconv(x[:, self.lbp_channels:, :, :])))

        if len(out_fuse) > 1:
            out = out_fuse[0] + out_fuse[1]
        else:
            out = out_fuse[0]

        out = self.rprelu(out)
        return out
   