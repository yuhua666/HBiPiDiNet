import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmdet.core import auto_fp16
from ..builder import NECKS
#可视化特征图
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys

cnt = 1
iii = 0
fpn = 0

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:, 0, :, :] * 0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap += feature_map[:, c, :, :]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps


def draw_feature_map(features, img_path, save_dir='feature_map', name=None):
    h = 800
    w = 800
    img = cv2.imread(img_path)
    global cnt
    global iii
    global fpn
    #plt.imshow(img)
    #cv2.imwrite(os.path.join("/root/IDa-Det-main/" +str(cnt)+"_img_"+ str(iii) +'.png'), img)
    if isinstance(features, torch.Tensor):
        for heat_maps in features:
            heat_maps = heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            for heatmap in heatmaps:
                # 这里的h,w指的是你想要把特征图resize成多大的尺寸
                heatmap = cv2.resize(heatmap, (h, w))
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = img + 0.3 * heatmap
                plt.imshow(superimposed_img)
                #plt.imshow(superimposed_img, cmap='jet')
                plt.show()
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            for heatmap in heatmaps:
                heatmap = cv2.resize(heatmap, (h, w))  # 将热力图的大小调整为与原始图像相同
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                #print(heatmap.shape)
                #print(heatmap)
                #print(type(heatmap))
                superimposed_img = img + 0.3 * heatmap
                # superimposed_img = heatmap
                plt.imshow(superimposed_img)
                #plt.imshow(superimposed_img, cmap='jet')
                plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                #global cnt
                cv2.imwrite(os.path.join("/root/IDa-Det-main/" +str(cnt)+"_p_"+ str(iii) +'.png'), superimposed_img)
                iii=iii+1
    if fpn == 0:
        fpn = 1
    else:
        cnt = cnt + 1
        iii = 0
        fpn = 0
                
@NECKS.register_module()
class FPN_REACT_3x3(nn.Module):
    """
    Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(FPN_REACT_3x3, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        #self.move0_ls = nn.ModuleList()
        #self.bn_ls = nn.ModuleList()
        #self.move1_ls = nn.ModuleList()
        self.prelu_ls = nn.ModuleList()
        #self.move2_ls = nn.ModuleList()
        
        #self.move0_fs = nn.ModuleList()
        #self.bn_fs = nn.ModuleList()
        #self.move1_fs = nn.ModuleList()
        self.prelu_fs = nn.ModuleList()
        #self.move2_fs = nn.ModuleList()
        
        for i in range(self.start_level, self.backbone_end_level):
            #move0_l = LearnableBias(in_channels[i])
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            #bn_l = nn.BatchNorm2d(out_channels)
            
            #move1_l = LearnableBias(out_channels)
            prelu_l = nn.PReLU(out_channels)
            #move2_l = LearnableBias(out_channels)
            
            #move0_f = LearnableBias(out_channels)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            #bn_f = nn.BatchNorm2d(out_channels)
            
            #move1_f = LearnableBias(out_channels)
            prelu_f = nn.PReLU(out_channels)
            #move2_f = LearnableBias(out_channels)

            
            #self.move0_ls.append(move0_l)
            self.lateral_convs.append(l_conv)
            #self.bn_ls.append(bn_l)
            #self.move1_ls.append(move1_l)
            self.prelu_ls.append(prelu_l)
            #self.move2_ls.append(move2_l)
            
            #self.move0_fs.append(move0_f)
            self.fpn_convs.append(fpn_conv)
            #self.bn_fs.append(bn_f)
            #self.move1_fs.append(move1_f)
            self.prelu_fs.append(prelu_f)
            #self.move2_fs.append(move2_f)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            self.prelu_ls[i](lateral_conv(inputs[i + self.start_level]))
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.prelu_fs[i](self.fpn_convs[i](laterals[i])) for i in range(used_backbone_levels)
        ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
                        
        #draw_feature_map(outs)
        
        # 可视化特征图
        '''for i in outs:
            feature_map = i #压缩成torch.Size([64, 55, 55])
            print("shape:")
            print(feature_map.shape)
            #以下4行，通过双线性插值的方式改变保存图像的大小
            #feature_map =feature_map.view(1,feature_map.shape[0],feature_map.shape[1],feature_map.shape[2])#(1,64,55,55)
            upsample = torch.nn.UpsamplingBilinear2d(size=(256,256))#这里进行调整大小
            feature_map = upsample(feature_map)
            feature_map = feature_map.view(feature_map.shape[1],feature_map.shape[2],feature_map.shape[3])
            
            feature_map_num = feature_map.shape[0]#返回通道数
            print("channel:")
            print(feature_map_num)
            row_num = int(np.ceil(np.sqrt(feature_map_num)))#8
            plt.figure()
            for index in range(1, feature_map_num + 1):#通过遍历的方式，将64个通道的tensor拿出
                plt.subplot(row_num, row_num, index)
                #plt.imshow(feature_map[index - 1], cmap='gray')#feature_map[0].shape=torch.Size([55, 55])
                #将上行代码替换成，可显示彩色 
                plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))#feature_map[0].shape=torch.Size([55, 55])
                plt.axis('off')
            global cnt
            cnt = cnt + 1
            print(cnt)
            plt.savefig(str(cnt)+"ourfpn.png")
        '''

        
        return tuple(outs)
