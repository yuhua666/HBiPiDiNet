# coding=UTF-8
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.autograd import Function
import math
from typing import Optional, List, Tuple

def binaryconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


def binaryconv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)

class HardBinaryConv(nn.Module):            #二值化卷积
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_chn
        self.out_channels = out_chn
        self.output_padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        #self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)        #直接把权重初始化成规定形状
        self.bin_act = BinaryActivation()

    def forward(self, x):
        #real_weights = self.weights.view(self.shape)
        x = self.bin_act(x)
        real_weights = self.weight-torch.mean(self.weight,dim=1,keepdim=True)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        #binary_weights_no_grad = torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y

class HardBinaryLinear(nn.Module):                  #与bert中quantizelinear作用相同
    def __init__(self, in_chn, out_chn):
        super(HardBinaryLinear, self).__init__()
        self.number_of_weights = in_chn * out_chn 
        self.shape = (out_chn, in_chn)
        #self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)
        self.bin_act = BinaryActivation()

    def forward(self, x):
        #real_weights = self.weights.view(self.shape)
        #real_weights = self.weight
        x = self.bin_act(x)
        real_weights = self.weight - torch.mean(self.weight, dim=1, keepdim=True)
        scaling_factor = torch.mean(abs(real_weights), dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        #binary_weights_no_grad = torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.linear(x, binary_weights)
        return y

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
        out = out_forward.detach() - out3.detach() + out3            #前向传递为sign激活，反向传播用approxsign的梯度近似sign的梯度

        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

'''
class LearnableBias(nn.Module):
    def __init__(self, out_chn, tot_win):
        super(LearnableBias, self).__init__()
        #self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)
        #第一个维度是Batchsize
        #第二个维度看是token数
        #第三个维度
        #self.bias = nn.Parameter(torch.zeros(tot_win,1,out_chn), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1,49,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
''' 
class ElasticQuantBinarizerSigned(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=True, init_method='default')
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)

        grad_scale = 1.0 / math.sqrt(input.numel()) if not Qp else 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            grad_alpha = ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        else:
            grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                    -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class ElasticQuantBinarizerUnsigned(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        Qn = 0
        Qp = 2 ** (num_bits) - 1
        if num_bits == 1:
            input_ = input
        else:
            min_val = input.min().item()
            input_ = input - min_val

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=False, init_method='default')
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)

        grad_scale = 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input_, alpha)
        ctx.other = grad_scale, Qn, Qp
        q_w = (input_ / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        if num_bits != 1:
            w_q = w_q + min_val
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big   # this is more cpu-friendly than torch.ones(input_.shape)
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None

class AlphaInit(nn.Parameter):
    def __init__(self, tensor):
        super(AlphaInit, self).__new__(nn.Parameter, data=tensor)
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, 'already initialized.'
        self.data.copy_(init_tensor)
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method='default'):
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if Qp == 0:
            Qp = 1.0
        if init_method == 'default':
            init_val = 2 * tensor.abs().mean() / math.sqrt(Qp) if symmetric \
                else 4 * tensor.abs().mean() / math.sqrt(Qp)
        elif init_method == 'uniform':
            init_val = 1./(2*Qp+1) if symmetric else 1./Qp

        self._initialize(init_val)


class BwnQuantizer(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        if layerwise:
            """
            #origial function
            s = input.size()
            m = input.norm(p=1).div(input.nelement())
            e = input.mean()
            result = (input-e).sign().mul(m.expand(s))
            """
            #add AdinBin
            s = input[0].nelement()
            beta_w = input.mean(1).view(-1,1)
            alpha_w = torch.sqrt(((input-beta_w)**2).sum(1)/s).view(-1,1)
            input = (input-beta_w)/alpha_w
            result = input.sign()*alpha_w + beta_w
        else:
            n = input[0].nelement()  # W of size axb, return a vector of  ax1
            s = input.size()
            m = input.norm(1, 1, keepdim=True).div(n)
            e = input.mean()
            result = (input-e).sign().mul(m.expand(s))

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def act_quant_fn(input, clip_val, num_bits, symmetric, quant_method, layerwise):
    if num_bits == 32:
        return input
    elif quant_method == "bwn" and num_bits == 1:
        quant_fn = BwnQuantizer
    elif quant_method == "elastic" and num_bits >= 1 and symmetric:
        quant_fn = ElasticQuantBinarizerSigned
    elif quant_method == "elastic" and num_bits >= 1 and not symmetric:
        quant_fn = ElasticQuantBinarizerUnsigned
    else:
        raise ValueError("Unknownquant_method")

    input = quant_fn.apply(input, clip_val, num_bits, layerwise)

    return input


def weight_quant_fn(weight,  clip_val,  num_bits,  symmetric, quant_method, layerwise):
    if num_bits == 32:
        return weight
    elif quant_method == "bwn" and num_bits == 1:
        quant_fn = BwnQuantizer
    else:
        raise ValueError("Unknown quant_method")

    weight = quant_fn.apply(weight, clip_val,  num_bits, layerwise)
    return weight


class QuantizeLinear(nn.Linear):

    def __init__(self, *kargs, clip_val=2.5, weight_bits=8, input_bits=8, learnable=False, symmetric=True,
                 weight_layerwise=True, input_layerwise=True, weight_quant_method="twn", input_quant_method="uniform",
                 **kwargs):
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.learnable = learnable
        self.symmetric = symmetric
        self.weight_layerwise = weight_layerwise
        self.input_layerwise = input_layerwise
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method
        self._build_weight_clip_val(weight_quant_method, learnable, init_val=clip_val)
        self._build_input_clip_val(input_quant_method, learnable, init_val=clip_val)
        self.move = LearnableBias(self.weight.shape[1])

    def _build_weight_clip_val(self, quant_method, learnable, init_val):
        if quant_method == 'uniform':
            # init_val = self.weight.mean().item() + 3 * self.weight.std().item()
            self.register_buffer('weight_clip_val', torch.tensor([-init_val, init_val]))
            if learnable:
                self.weight_clip_val = nn.Parameter(self.weight_clip_val)
        elif quant_method == 'elastic':
            assert learnable, 'Elastic method must use leranable step size!'
            self.weight_clip_val = AlphaInit(torch.tensor(1.0)) # stepsize will be initialized in the first quantization
        else:
            self.register_buffer('weight_clip_val', None)

    def _build_input_clip_val(self, quant_method, learnable, init_val):
        if quant_method == 'uniform':
            self.register_buffer('input_clip_val', torch.tensor([-init_val, init_val]))
            if learnable:
                self.input_clip_val = nn.Parameter(self.input_clip_val)
        elif quant_method == 'elastic' or quant_method == 'bwn':
            assert learnable, 'Elastic method must use leranable step size!'
            self.input_clip_val = AlphaInit(torch.tensor(1.0))  # stepsize will be initialized in the first quantization
        else:
            self.register_buffer('input_clip_val', None)

    def forward(self, input):
        # quantize weight
        weight = weight_quant_fn(self.weight, self.weight_clip_val, num_bits=self.weight_bits, symmetric=self.symmetric,
                                 quant_method=self.weight_quant_method, layerwise=self.weight_layerwise)
        # quantize input
        input = self.move(input)
        input = act_quant_fn(input, self.input_clip_val, num_bits=self.input_bits, symmetric=self.symmetric,
                             quant_method=self.input_quant_method, layerwise=self.input_layerwise)
        out = nn.functional.linear(input, weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class QuantizeEmbedding(nn.Embedding):

    def __init__(self, *kargs, clip_val=2.5, weight_bits=8, learnable=False, symmetric=True,
                 embed_layerwise=False, weight_quant_method="twn", **kwargs):
        super(QuantizeEmbedding, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.learnable = learnable
        self.symmetric = symmetric
        self.embed_layerwise = embed_layerwise
        self.weight_quant_method = weight_quant_method
        self._build_embed_clip_val(weight_quant_method, learnable, init_val=clip_val)

    def _build_embed_clip_val(self, quant_method, learnable, init_val):
        if quant_method == 'uniform':
            self.register_buffer('embed_clip_val', torch.tensor([-init_val, init_val]))
            if learnable:
                self.embed_clip_val = nn.Parameter(self.embed_clip_val)
        elif quant_method == 'elastic':
            assert learnable, 'Elastic method must use leranable step size!'
            self.embed_clip_val = AlphaInit(torch.tensor(1.0)) # stepsize will be initialized in the first quantization
        else:
            self.register_buffer('embed_clip_val', None)

    def forward(self, input):
        weight = weight_quant_fn(self.weight, self.embed_clip_val, num_bits=self.weight_bits, symmetric=self.symmetric,
                                 quant_method=self.weight_quant_method, layerwise=self.embed_layerwise)

        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

        return out

class BinaryQuantize(Function):
    '''
        binary quantize function, from IR-Net
        (https://github.com/htqin/IR-Net/blob/master/CIFAR-10/ResNet20/1w1a/modules/binaryfunction.py)
    ''' 
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        k, t = k.cuda(), t.cuda() 
        grad_input = k * t * (1-torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None

    def forward(self, inputs):
        if self.a_bit==1:
            inputs = self.binary_a(inputs) 

        if self.w_bit==1:
            w = self.weight 
            beta_w = w.mean((1,2,3)).view(-1,1,1,1)
            alpha_w = torch.sqrt(((w-beta_w)**2).sum((1,2,3))/self.filter_size).view(-1,1,1,1)

            w = (w - beta_w)/alpha_w 
            wb = BinaryQuantize().apply(w, self.k, self.t)
            weight = wb * alpha_w + beta_w
        else: 
            weight = self.weight
        
        output = F.conv2d(inputs, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output

class SignTwoOrders(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        grad_input_wrt_output2 = torch.zeros_like(grad_output)
        ge0_lt1_mask = input.ge(0) & input.lt(1)
        grad_input_wrt_output2 = torch.where(ge0_lt1_mask, (2 - 2 * input), grad_input_wrt_output2)
        gen1_lt0_mask = input.ge(-1) & input.lt(0)
        grad_input_wrt_output2 = torch.where(gen1_lt0_mask, (2 + 2 * input), grad_input_wrt_output2)
        grad_input = grad_input_wrt_output2 * grad_output

        return grad_input


class SignSTE(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        mask = input.ge(-1) & input.le(1)
        grad_input = torch.where(mask, grad_output, torch.zeros_like(grad_output))
        return grad_input


class SignSTEWeight(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_empty(grad_output.size())
        grad_input.copy_(grad_output)
        return grad_input


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, weight_magnitude_aware=True, activation_value_aware=True,
                 **kwargs):
        super(BinarizeConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.weight_magnitude_aware = weight_magnitude_aware
        self.activation_value_aware = activation_value_aware

    def forward(self, input):
        if self.activation_value_aware:
            input = SignTwoOrders.apply(input)
        else:
            input = SignSTE.apply(input)

        subed_weight = self.weight
        if self.weight_magnitude_aware:
            self.weight_bin_tensor = subed_weight.abs(). \
                                         mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True) \
                                     * SignSTEWeight.apply(subed_weight)
        else:
            self.weight_bin_tensor = SignSTEWeight.apply(subed_weight)
        self.weight_bin_tensor.requires_grad_()
        #print("xbinary:")
        #print(input)
        input = F.pad(input, (self.padding[0], self.padding[0], self.padding[1], self.padding[1]),
                      mode='constant', value=-1)
        out = F.conv2d(input, self.weight_bin_tensor, self.bias, self.stride, 0, self.dilation, self.groups)
        return out


class BinarizeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BinarizeLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        input = SignTwoOrders.apply(input)

        self.weight_bin_tensor = SignSTEWeight.apply(self.weight)
        self.weight_bin_tensor.requires_grad_()

        out = F.linear(input, self.weight_bin_tensor, self.bias)

        return out


def myid(x):
    return x


class BinBlock(nn.Module):
    def __init__(self, inplanes, planes, res_func=myid, **kwargs):
        super(BinBlock, self).__init__()
        self.conv = BinarizeConv2d(inplanes, planes, **kwargs)
        self.bn = nn.BatchNorm2d(planes)
        self.res_func = res_func

    def forward(self, input):
        if self.res_func is not None:
            residual = self.res_func(input)
        out = self.conv(input)
        out = self.bn(out)
        if self.res_func is not None:
            out += residual
        return out

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

        y = out_forward.detach() + out - out.detach()
        return y

class LBPConv(nn.Module):
    """
    local binary convolution
    """
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

class RSign(nn.Module):

    def __init__(self, chn, bound=1.2, binary=True):
        super(RSign, self).__init__()
        self.bound = bound
        self.binary = binary
        self.beta = nn.Parameter(torch.zeros(1, chn, 1, 1))

    def reset_state(self, bound, binary):
        self.bound = bound
        self.binary = binary

    def forward(self, x):
        if not self.binary:
            return x

        x = x + self.beta.expand_as(x)
        if not self.training:
            return torch.sign(x)

        #print('activation before sign: ', x, flush=True)
        out = torch.clamp(x, -self.bound, self.bound)
        out_forward = torch.sign(x)

        y = out_forward.detach() + out - out.detach()
        return y

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
    
from torch.nn.modules.conv import _ConvNd
from itertools import repeat
from torch.nn import Parameter
import collections

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


'''
if __name__ == '__main__':
    bin_ac =BinaryActivation()
    x = torch.rand(512,512,512)-0.5
    BinarizeConv2d(x,512,512)
    #print(bin_ac(x))
    #print(bin_ac(bin_ac(x)))
    #print(bin_ac(x)==bin_ac(bin_ac(x)))
'''