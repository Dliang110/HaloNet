
import torch
from torch import Tensor
import torch.nn as nn
#from .._internally_replaced_utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
import torch
from halonet_pytorch import HaloAttention



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    #TODO:
    def __init__():
        self.init = 'ToDO'


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block_size: int =8,
        halo_size: int = 4,
        dim_head: int = 32,
        heads: int =4,
        rv: int = 1,
        rb: int =1,
        verbose:  bool=False
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        # self.bn2 = norm_layer(width)

        self.attn = HaloAttention(
            dim =  width ,         # dimension of feature map
            block_size = block_size,    # neighborhood block size (feature map must be divisible by this)
            halo_size = halo_size,     # halo size (block receptive field)
            dim_head = round( dim_head * rv),   #* rv   # dimension of each head
            heads = heads          # number of attention heads
        )
        #print(' self.expansion    ', self.expansion )
        self.conv3 = conv1x1( round(width ), round(planes * self.expansion * rb))
        self.bn3 = norm_layer( round(planes * self.expansion * rb))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.verbose = verbose

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        #print('ID sahpe ', identity.shape)
        #print('X SHAPE ', x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.verbose:
            print('out conV1 shape ', out.shape)
        # out = self.conv2(out)
        # out = self.bn2(out)
        out  =  self.attn(out)
        if self.verbose:
            print('\n\n',
              'out attn shape ', out.shape)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.verbose:
            print('out Conv2 shape ', out.shape)
        #print('Identity shape  ', identity.shape)
        #print('Downsample ', self.downsample)
        if self.downsample is not None:
            identity = self.downsample(x)
            #print('Identity shape after downsampling ', identity.shape)
        out += identity
        out = self.relu(out)
        if self.verbose:
            print('End of layer ! \n\n')

        return out


class HaloNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: dict,
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        #block_size:int = 8,
        #halo_size: int = 3,
        dim_head: int = 16,
    ) -> None:
        super(HaloNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.block_size = layers['block_size']
        self.halo_size = layers['halo_size']
        self.df = layers['df']
        self.rv =   layers['rv']
        self.rb =   layers['rb']
        self.inplanes = 64
        self.dilation = 1
        self.dim_head = dim_head
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers['stage0'][0], block_size= self.block_size,
                                     halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage0'][1],
                                      rv=self.rv, rb=self.rb , verbose= False )
        self.layer2 = self._make_layer(block, 128, layers['stage1'][0], block_size= self.block_size,
                                     halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage1'][1], stride=1,
                                       dilate=replace_stride_with_dilation[0],
                                        rv=self.rv, rb=self.rb, verbose= False )
        self.layer3 = self._make_layer(block, 256, layers['stage2'][0], block_size= self.block_size,
                                     halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage2'][1], stride=1,
                                       dilate=replace_stride_with_dilation[0],
                                        rv=self.rv, rb=self.rb )
        self.layer4 = self._make_layer(block, 512, layers['stage3'][0], block_size= self.block_size,
                                     halo_size=self.halo_size, dim_head= self.dim_head, heads= layers['stage3'][1], stride=1,
                                     dilate=replace_stride_with_dilation[0],
                                      rv=self.rv, rb=self.rb )
        if self.df != -1:
            ## TODO:  Implement
            self.convf = conv1x1(round(512 * block.expansion * self.rb), self.df)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.df, num_classes)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(round(512 * block.expansion * self.rb), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, block_size:int =8, halo_size: int = 10, dim_head: int = 16,
                     heads: int = 8, rv:float = 1, rb:float = 1 , verbose: bool=False) -> nn.Sequential:

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion or self.rb != 1:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, round(planes * block.expansion * self.rb), stride),
                norm_layer(round(planes * block.expansion * self.rb)),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, block_size, halo_size,
                            dim_head, heads, rv, rb, verbose))
        self.inplanes = round(planes * block.expansion * self.rb)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, block_size = block_size, halo_size =halo_size,
                                 dim_head= dim_head, heads =heads, rv=rv, rb=rb, verbose=verbose))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print('BF layer 1: ', x.shape)
        x = self.layer1(x)
        #print('\n\n+++++++++++++++++++++++++++++\n\nlayer1 x shape ', x.shape)
        x = self.layer2(x)
        #print('layer2 x shape ', x.shape)
        #print('\n\n+++++++++++++++++++++++++++++\n\nlayer2 x shape ', x.shape)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.df  != -1 :
            x = self.convf(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _halonet(
    arch: str,
    block: Type[Union[ Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> HaloNet:
    model = HaloNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def halonetB0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> HaloNet:
    r"""
    """
    # (arg1: number of block by stage, arg2: nb of heads, args3: rv, args4: rb, args5: df )
    return _halonet('halonetB0', Bottleneck, {'block_size':8, 'halo_size':3,'stage0':(3,4), 'stage1':(3,8),
                                            'stage2':(7,8), 'stage3':(3,8),
                                            'rv':1, 'rb':0.5, 'df':-1}, pretrained, progress,
                   **kwargs)
def halonetB1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> HaloNet:
    r"""h
    """
    # (arg1: number of block by stage, arg2: nb of heads, args3: rv, args4: rb, args5: df )
    return _halonet('halonetB1', Bottleneck, {'block_size':8, 'halo_size':3,'stage0':(3,4), 'stage1':(3,8),
                                            'stage2':(10,8), 'stage3':(3,8),
                                            'rv':1, 'rb':1, 'df':-1}, pretrained, progress,
                   **kwargs)
def halonetB2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> HaloNet:
    r"""h
    """
    # (arg1: number of block by stage, arg2: nb of heads, args3: rv, args4: rb, args5: df )
    return _halonet('halonetB2', Bottleneck, {'block_size':8, 'halo_size':3,'stage0':(3,4), 'stage1':(3,8),
                                            'stage2':(11,8), 'stage3':(3,8),
                                            'rv':1, 'rb':1.25, 'df':-1}, pretrained, progress,**kwargs)
def halonetB3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> HaloNet:
    r"""h
    """
    # (arg1: number of block by stage, arg2: nb of heads, args3: rv, args4: rb, args5: df )
    return _halonet('halonetB3', Bottleneck, {'block_size':10, 'halo_size':3,'stage0':(3,4), 'stage1':(3,8),
                                            'stage2':(12,8), 'stage3':(3,8),
                                            'rv':1, 'rb':1.5, 'df':1024}, pretrained, progress,**kwargs)
def halonetB4(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> HaloNet:
    r"""h
    """
    # (arg1: number of block by stage, arg2: nb of heads, args3: rv, args4: rb, args5: df )
    return _halonet('halonetB4', Bottleneck, {'block_size':12, 'halo_size':2,'stage0':(3,4), 'stage1':(3,8),
                                            'stage2':(12,8), 'stage3':(3,8),
                                            'rv':1, 'rb':3, 'df':1280}, pretrained, progress,**kwargs)
def halonetB5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> HaloNet:
    r"""h
    """
    # (arg1: number of block by stage, arg2: nb of heads, args3: rv, args4: rb, args5: df )
    return _halonet('halonetB5', Bottleneck, {'block_size':14, 'halo_size':2,'stage0':(3,4), 'stage1':(3,8),
                                            'stage2':(23,8), 'stage3':(3,8),
                                            'rv':2.5, 'rb':2, 'df':1536}, pretrained, progress,**kwargs)
def halonetB6(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> HaloNet:
    r"""
    """
    # (arg1: number of block by stage, arg2: nb of heads, args3: rv, args4: rb, args5: df )
    return _halonet('halonetB5', Bottleneck, {'block_size':8, 'halo_size':4,'stage0':(3,4), 'stage1':(3,8),
                                            'stage2':(24,8), 'stage3':(3,8),
                                            'rv':3, 'rb':2.75, 'df':1536}, pretrained, progress,**kwargs)
def halonetB7(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> HaloNet:
    r"""h
    """
    # (arg1: number of block by stage, arg2: nb of heads, args3: rv, args4: rb, args5: df )
    return _halonet('halonetB7', Bottleneck, {'block_size':10, 'halo_size':3,'stage0':(3,4), 'stage1':(3,8),
                                            'stage2':(24,8), 'stage3':(3,8),
                                            'rv':4, 'rb':3.50, 'df':2048}, pretrained, progress,**kwargs)
if __name__ == '__main__':
    from torchsummary import summary
    model = halonetB7()
    #print(model)
    summary(model, (3,600,600))
