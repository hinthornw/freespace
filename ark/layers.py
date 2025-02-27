import os
import numpy as np
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import ones, cat
import cv2
import time
from layers import *

#from .modules.utils import _single, _pair, _triple
#from . import _functions
from numbers import Integral
from torch._thnn import type2backend



def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding,dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dil=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride) #, dilation=dil) #temp-conv1 always dil=1
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dil)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dil=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, dilation=dil, stride=stride,
                               padding=dil, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        '''First 1x1 to fewer channels, then 3x3 conv, then 1x1 back to original space.
        May include downsampling'''
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class DUCSmooth(nn.Module):
    def __init__(self, inplanes, planes,r,kernel_size=1,res=False, f='ELU', grouped=True):
        super(DUCSmooth, self).__init__()
        self.DUC = DUC(inplanes, planes,r,kernel_size, res, grouped=grouped)
        self.conv1 =  nn.Conv2d(planes, planes,3,1,1,bias=False)
        if f=='ELU':
            self.f = nn.ELU(inplace=True)
        elif f=='ID':
            self.f = Identity()
        else:
            self.f = nn.Sigmoid()
    def forward(self, x):
        x = self.DUC(x)
        x = self.conv1(x)
        x = self.f(x)
        return x
        

class DUC(nn.Module):
    ''' Dense Upsampling Convolution
    '''
    def __init__(self, inplanes, l, r, kernel_size=3, res=False, grouped=True):
        super(DUC, self).__init__()
        L = l * r * r
        padding = 1
        if kernel_size == 1:
            padding = 0 
        if not grouped:
            self.conv1_ = nn.Conv2d(inplanes,L, kernel_size,1,padding,bias=False)
        else:
            self.conv1_ = nn.Conv2d(inplanes,L, kernel_size,1,padding,bias=False, groups=inplanes)
        self.res = res
        if res:
            self.residual = nn.Conv2d(inplanes, L, kernel_size=1, padding=0, bias=False, groups=inplanes)
        #self.conv2_ = nn.Conv2d(L, L, groups=l, kernel_size=1, padding=0, stride=1)
        # Ithink conv2 should have kernel_size=1
        self.f = nn.ELU(inplace=False)
        self.duc = nn.PixelShuffle(r)
    
    def forward(self, x):
        out = self.f(self.conv1_(x))
        if self.res:
            out = out + self.residual(x)
        out = self.duc(out)
        return out

class LeapBlock(nn.Module):

    def __init__(self, inplanes, exp, down,smooth=True, numsumm = 1, numsmooth=1, res=(False,False)):
        super(LeapBlock, self).__init__() 
        self.planes = inplanes*exp
        self.shuff = ShuffleBlock(inplanes,self.planes, down,smooth=smooth,res=res[0])
        summ = []
        for  i in range(numsumm):
            summ.append(SummaryBlock(self.planes, self.planes))
        self.summ = nn.Sequential(*summ)
        self.up = DUC(self.planes,inplanes, down,3, res[1], grouped=True)  
        smooth = []
        for i in range(numsmooth):
            smooth.append(nn.Conv2d(inplanes, inplanes,3,1,1,bias=False))
        self.smooth = nn.Sequential(*smooth)
        self.f = nn.ELU(inplace=True) 
    def forward(self, x):
        res = x
        x = self.shuff(x)
        x = self.summ(x)
        x = self.up(x)
        x = x + res
        x = self.smooth(x)
        return x

###########################################################################################
# DenseNet Stuff 
#
###########################################################################################
class DenseLayer(nn.Module):
    def __init__(self, inplanes, gr, dilation):
        super(DenseLayer, self).__init__()
       # self.conv1 = nn.Conv2d(inplanes, gr*4,1,1,0,bias=False)
      #  self.conv2 = nn.Conv2d(gr*4,gr,3,1,1,bias=False)
        self.conv2 = nn.Conv2d(inplanes,gr,3,1,dilation,dilation,bias=False)
      #  self.bn1 = nn.BatchNorm2d(inplanes)
        #self.bn2 = nn.BatchNorm2d(gr*4)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.f = nn.ReLU(inplace=True)
    def forward(self, x):
       # x = self.conv1(self.f(self.bn1(x)))
        x = self.conv2(self.f(self.bn2(x)))
        return x
    
class TransitionUp(nn.Module):
    def __init__(self, inplanes, planes, res=False):
        super(TransitionUp, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.f = nn.ReLU(inplace=True)
        self.up = nn.ConvTranspose2d(planes, planes, 3, 2, 1,output_padding=1, bias=False)
    def forward(self, x):
        return self.up(self.f(self.bn1(x)))
 
class TransitionLayer(nn.Module):
    def __init__(self, inplanes, planes, downscale_factor=2, res=False): 
        super(TransitionLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.f = nn.ReLU()
        self.conv1 = nn.Conv2d(inplanes, planes, 3,1,1,bias=False)
#        self.down = nn.AvgPool2d(2, stride=2)
        
        self.down = nn.Conv2d(planes,planes,3,2,1,bias=False)
    def forward(self, x):
        return self.down(self.f(self.bn2(self.conv1(self.f(self.bn1(x))))))
#    def __init__(self, inplanes, planes, downscale_factor=2, res=False): 
#        super(TransitionLayer, self).__init__()
#        self.f = nn.ELU(inplace=False)
#        self.conv1 = nn.Conv2d(inplanes, planes, 1,1,0,bias=False)
#        self.down = DenseDownsamplingBlock(planes, planes, downscale_factor=downscale_factor, res=res)
#    def forward(self, x):
#        x = self.conv1(self.f(x))
#        
#        return self.down(x) 

class DenseBlock(nn.Module):
    def __init__(self, inplanes, num_layers,gr,  dilations, residual=True):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.planes = inplanes
        self.res = residual
        for i in range(num_layers):
            self.layers.append(DenseLayer(self.planes, gr, dilations[i]))
            self.planes += gr
    def forward(self, x):
        output = [] 
        for i, l in enumerate(self.layers):
            out = l(x)
            output.append(out)
            x = cat([x, out], 1) 
        if self.res:
            return x
        else:
            return cat(output, 1)


class Diff(nn.Module):
    def __init__(self):
        super(Diff, self).__init__()
    def forward(self, x, y):            
        return x - y

class L1VertMask(nn.Module):
    def __init__(self,inb, inw,inh,num_classes=2, deviceids=0):
        super(L1VertMask, self).__init__()
        self.b = inb
        self.inw, self.inh = inw, inh
        self.ch = num_classes
        self.thresh = nn.Threshold(1.0,1.0)
        self.expand = nn.ReplicationPad2d((0,0,inh-1,0))
        self.mask = Variable((torch.arange(0, inh*inw)%inh).view(inb,1,inh,inw).permute(0,1,3,2).contiguous()).cuda(deviceids)
#        self.mask.requires_grad = False

    def forward(self, x, param):
        params = self.expand(param) 
        self.mask = self.mask.detach()
        a = self.mask - params
        b = - a
        a = self.thresh(a)
        b = self.thresh(b)
        #print b[0,0,:,30] 
        mask = cat([a,b],1)
        out = x*mask
#        print "OUT:{}".format(out)
        return x*mask
                         
       


###########################################################################################
# Below are recurrent Layers
#
###########################################################################################

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x




class Gate(nn.Module):
    def __init__(self, sig_c, sup_c):
        self.suptosig = nn.Conv2d(sup_c, sig_c,3,1,1,bias=False)
        self.sigtosig = nn.Conv2d(sup_c, sig_c, 3,1,1,bias=False)
        self.f = nn.Sigmoid()


    def forward(self, signal, sup):
        out = self.f(self.sigtosig(signal) - self.suptosig(sup))
        return out


class MultiLayerRecurrent(nn.Module):
    def __init__(self, c):
        super(MultiLayerRecurrent, self).__init__()
        self.convhidden = nn.Conv2d(c,c,3,1,1,bias=False)
        self.convhidden2 = nn.Conv2d(c,c,3,1,1,bias=False)
        self.convDiff = nn.Conv2d(c,c,3,1,1,bias=False)
        self.convDiff2 = nn.Conv2d(c,c,3,1,1,bias=False)
        self.convCurr = nn.Conv2d(c,c,3,1,1,bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.bn2 = nn.BatchNorm2d(c)
        self.f = nn.ReLU(inplace=True)


    def forward(self, curr, prev, h):
        h  = self.f(self.bn1(self.convDiff2(self.convDiff(curr - prev)) + self.convhidden(h))) 
        out = self.f(self.bn2(self.convCurr(curr) + self.convhidden2(h)))
        return out, h

class ConvElman(nn.Module):
    def __init__(self, c, hidden_size, kernel_size, padding=1, layers=1, rel='add', f=None):
        #TODO: Implement multiple layers.
        super(ConvElman, self).__init__()
        if rel == 'add' or rel == 'sub':
            self.rel = 1
        else:
            self.rel = 0
        self.ith = nn.Conv2d(c, hidden_size, kernel_size=kernel_size, padding=padding, bias=False)
        self.hth = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding,bias=False)
        self.hto = nn.Conv2d(hidden_size, c, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(c)
        if f == 'id':
            self.g = Identity()
            self.f = Identity()
        elif f == 'gatedELU':
            self.g = nn.Sigmoid()
            self.f = nn.ELU(inplace=True)
        elif f == 'gatedID':
            self.g = Identity()
            self.f = nn.ELU(inplace=True)
        else:
            self.g = Identity()
            self.f = nn.ELU(inplace=True)

    def forward(self, x, h):
        if self.rel==1:
            h_t = self.f(self.ith(x) + self.g(self.hth(h)))
        else:
            h_t = self.f(self.ith(x)*self.g(self.hth(h)))
        y_t = self.hto(h_t)
        y_t = self.f(self.bn(y_t))
        return y_t, h_t



class ConvGRU(nn.Module):
    def __init__(self, c, hidden_size, kernel_size, padding=1, layers=1):
        #TODO: Implement multiple layers.
        super(ConvGRU, self).__init__()
        self.itz = nn.Conv2d(c, hidden_size, kernel_size=kernel_size, padding=padding, bias=False)
        self.itr = nn.Conv2d(c, hidden_size, kernel_size=kernel_size, padding=padding, bias=False)
        self.ith = nn.Conv2d(c, hidden_size, kernel_size=kernel_size, padding=padding, bias=False)


        self.htz = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding, bias=False)
        self.htr = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding, bias=False)
        self.hth = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding, bias=False)
        self.f = nn.ELU(inplace=True) # Note I am replacing the tanh with an ELU function...
        self.sig = nn.Sigmoid()
        self.convout = nn.Conv2d(hidden_size, c, kernel_size=1, padding=0)

    def forward(self, x, h):
        z_t = self.sig(self.itz(x) + self.htz(h)) 
        r_t = self.sig(self.itr(x) + self.htr(h))
        h_t = z_t * h + (1-z_t) * self.f(self.ith(x) + self.hth(r_t*h))
        y_t = self.f(self.convout(h_t)) #For dimensionality concerns    
        return y_t, h_t


class RecLayer(nn.Module):
    def __init__(self, c, rnnlayer, kernel_size=3, padding=1):
        super(RecLayer, self).__init__()
        self.convdown = nn.Conv2d(c, c//4, kernel_size=1, padding=0, stride=1, bias=False)
        self.rnn = rnnlayer(c//4, c//2, kernel_size=kernel_size, padding=1)
        self.convout = nn.Conv2d(c+c//4, c, kernel_size=1, padding=0, stride=1, bias=False)
        self.f = nn.ELU(inplace=True)
    
    def forward(self, x, h):
        residual = x
        out = self.convdown(x)
        out, h_t = self.rnn(out, h)
        out = cat([residual, out], 1)
        out = self.convout(out)
        out = self.f(out)
        return out, h_t

class RecLayerBasic(nn.Module):
    def __init__(self, c, rnnlayer, kernel_size=3, padding=1):
        super(RecLayerBasic, self).__init__()
        self.rnn = rnnlayer(c, 2*c, kernel_size=kernel_size, padding=1)
        self.convout = nn.Conv2d(2*c, c, kernel_size=3, padding=1, stride=1, bias=False)
        self.f = nn.ELU(inplace=True)
    
    def forward(self, x, h):
        residual = x
        out, h_t = self.rnn(x, h)
        out = cat([residual, out], 1)
        out = self.convout(out)
        out = self.f(out)
        return out, h_t


class BasicBlockx(nn.Module):
    expansion = 1

    #def __init__(self, inplanes, planes, stride=1, dil=1, downsample=None):
    def __init__(self, inplanes, planes, stride=1, dil=(1,1), downsample=None, groups=2):
        super(BasicBlockx, self).__init__() 
        self.conv1 = conv3x3(inplanes, planes, stride=stride, padding=dil[0], dilation=dil[0])
        #self.bn1 = nn.BatchNorm2d(planes)
        self.f = nn.ReLU(inplace=False) #nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dil[1], dilation=dil[1])#, groups=planes/4)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(self.f(self.bn1(x)))
        out = self.conv2(self.f(self.bn2(out)))
        if self.downsample is not None:
            residual = self.downsample(x)
        return residual + out

class DenseFusion(nn.Module):
    def __init__(self, inplanes, planes):
        super(DenseFusion, self).__init__()
        self.fuse1  = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1,bias=False)
        self.bnfuse = nn.BatchNorm2d(planes)
        self.single = nn.Conv2d(planes, planes, kernel_size=3, padding=1,bias=False)
        self.doubdil= nn.Conv2d(planes, planes, kernel_size=3, padding=2,dilation=2,bias=False)
        self.triple = nn.Conv2d(planes, planes, kernel_size=3, padding=3,dilation=3,bias=False)
        self.penta  = nn.Conv2d(planes, planes, kernel_size=3, padding=5,dilation=5,bias=False)
        self.nova   = nn.Conv2d(planes, planes, kernel_size=3, padding=9,dilation=9,bias=False)
        self.f = nn.ELU(inplace=True)
        self.bnfused = nn.BatchNorm2d(planes)
        self.smooth1 = nn.Conv2d(planes, inplanes, kernel_size=3, padding=1,bias=False)
        self.bnout = nn.BatchNorm2d(inplanes)
        self.smooth2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1,bias=False)

    def forward(self, x):
        residual = x 
        fused = self.f(self.bnfuse(self.fuse1(x))) 
        single = self.f(self.single(fused))
        double = self.f(self.doubdil(fused))
        triple = self.f(self.triple(fused))
        penta  = self.f(self.penta(fused))
        nova   = self.f(self.nova(fused))
        out = fused + single + double + triple + penta + nova
        out = self.bnfused(out)
        out = self.f(self.smooth1(out))
        out = self.smooth2(out)
        return out


class DenserFusion(nn.Module):
    def __init__(self, inplanes, planes):
        super(DenserFusion, self).__init__()
        self.fuse1  = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1,bias=False)
        self.bnfuse = nn.BatchNorm2d(planes)
        self.vert = nn.Conv2d(planes, planes, kernel_size=(7,1), padding=(3,0), bias=False)
        self.horiz = nn.Conv2d(planes, planes, kernel_size=(1,7), padding=(0,3), bias=False)
        self.single = nn.Conv2d(planes, planes, kernel_size=3, padding=1,bias=False)
        self.doubdil= nn.Conv2d(planes, planes, kernel_size=3, padding=2,dilation=2,bias=False)
        self.triple = nn.Conv2d(planes, planes, kernel_size=3, padding=3,dilation=3,bias=False)
        self.f = nn.ELU(inplace=True)
        self.bnfused = nn.BatchNorm2d(planes)
        self.smooth1 = nn.Conv2d(planes, inplanes, kernel_size=3, dilation=2, padding=2,bias=False)
        self.bnout = nn.BatchNorm2d(inplanes)

    def forward(self, x):
        residual = x 
        fused = self.f(self.bnfuse(self.fuse1(x))) 
        bull = self.f(self.horiz(fused) + self.vert(fused))
        single = self.f(self.single(bull))
        double = self.f(self.doubdil(bull))
        triple = self.f(self.triple(bull))
        out = fused + single + double + triple 
        out = self.bnfused(out)
        out = residual + self.f(self.smooth1(out))
        return out


class HDInput(nn.Module):
    def __init__(self, inplanes, planes, downscale=2, insize=896):
        super(HDInput, self).__init__() 
        
        self.DDC = DDC(inplanes,downscale, insize) 
        self.convin= nn.Conv2d(inplanes*downscale**2, planes/16, kernel_size=3, stride=2, dilation=1, padding=1, bias=False)
        self.f = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(planes/16)
        self.conv1 = nn.Conv2d(planes/16, planes/8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(planes/8)
        self.conv2 = nn.Conv2d(planes/8, planes/2, kernel_size=3, stride=2, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(planes/2)
        self.conv3 = nn.Conv2d(planes/2, planes, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn3 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.DDC(x)
        out = self.bn1(self.convin(out)) 
        out = self.bn1b(self.conv1(out))
        out = self.bn2(self.conv2(out))
        out = self.f(self.bn3(self.conv3(out)))
        return out

class DDC(nn.Module):
    def __init__(self, inplanes, downscale_factor):
        """Rearranges elements in a tensor of shape ``[*, C, H, W]`` to a
        tensor of shape ``[*, r^2*C, H/r, W/r]``.
        """
        super(DDC, self).__init__()
#        self.bs = 1
        self.inplanes = inplanes
        self.planes = inplanes*(downscale_factor**2) 
        self.scale = downscale_factor 
       # self.outh, self.outw = insize//downscale_factor, insize//downscale_factor 

    def forward(self, input):
        b,c,h,w = input.size()
        self.outh = h//self.scale
        self.outw = self.outh
        input_view = input.contiguous().view(b,c,self.outh,self.scale,self.outw,self.scale) 
        return input_view.permute(0,1,3,5,2,4).contiguous().view(b,
                                            self.planes, self.outh, self.outw)
class ShuffleBlock(nn.Module):

    def __init__(self, inplanes, planes,downscale_factor=3, smooth=True, res=False):
        super(ShuffleBlock, self).__init__()
        self.scale = downscale_factor
        self.down = DenseDownsamplingBlock(inplanes, planes, downscale_factor=downscale_factor, res=res)
        self.smooth = smooth
        if smooth:
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, dilation=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, dilation=1, bias=False)
        self.f = nn.ELU(inplace=True) #nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.down(x) # downsample 
        if self.smooth:
            out = self.bn1(out)
            residual = out 
            out = self.conv1(self.f(out))
            out = self.conv2(self.f(self.bn2(out))) 
            out = out + residual 
        return out

class SummaryBlock(nn.Module):
    def __init__(self, inplanes, planes, dil=(1,1)):
        super(SummaryBlock, self).__init__()
        self.f1 = nn.ELU(inplace=False) #nn.ReLU(inplace=True)
        self.f2 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, padding=dil[0], dilation=dil[0])
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, padding=dil[1], dilation=dil[1])#, groups=planes/4)
        self.dynamic = inplanes != planes
        if self.dynamic:
            self.manip = nn.Conv2d(inplanes, planes,1,1,0,bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(self.f1(x))
        out = self.conv2(self.f2(self.bn2(out)))
        if self.dynamic:
            residual = self.manip(residual)
        out = out + residual #+ out1
        return out
class GroupedBlock(nn.Module):
    def __init__(self, inplanes, crate, groups, kernel_size=3, dilations=0, res=False):
        super(GroupedBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes/crate, 1,1,0,bias=False)
        self.conv2 = nn.Conv2d(inplanes/crate, inplanes/crate,kernel_size,1,1,groups=groups,bias=False)
        self.conv3 = nn.Conv2d(inplanes/crate, inplanes, 1,1,0,bias=False)
        self.f = nn.ELU(inplace=True)
        
    def forward(self, x):
        if self.res:
            residual = x
        x = self.conv1(self.f(x))
        x = self.conv2(x)
        x = self.conv3(self.f(x))
        if self.res:
            x = x + residual
        return x
class GracefulDownsampling(nn.Module):
    def __init__(self, inplanes, planes):
        super(GracefulDownsampling, self).__init__()
        self.bump = 9
        self.exp = inplanes * self.bump
        #outsize = (sizein-1)*stride - 2(padding) + kernel + outpadding
        self.trans = nn.ConvTranspose2d(inplanes, inplanes,3,2, padding=1,output_padding=1, bias=False) 
        self.DDC = DDC(inplanes, 3)
        self.conv1_v1 = nn.Conv2d(self.exp, self.exp, 1,1,0,bias=False, groups=9)
        self.conv2_v1 = nn.Conv2d(self.exp, planes, 3,1,1,bias=False)
        #self.f = nn.ELU(inplace=True)
        self.f = nn.ELU()

    def forward(self, x):
        x = self.f(x)
        x = self.trans(x)
        x = self.DDC(x)
        x = self.conv1_v1(x)
        x = self.f(self.conv2_v1(x))
        return x

class DenseDownsamplingBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, downscale_factor=3, res=True):
        super(DenseDownsamplingBlock, self).__init__()
        self.bump = downscale_factor**2
        self.exp = self.bump * inplanes 
        self.DDC = DDC(inplanes, downscale_factor) 
        #self.conv2_v1 = nn.Conv2d(inplanes,inplanes,3,1,2,2,bias=False)
        #self.conv1_up_v1 = nn.Conv2d(inplanes, inplanes, 3,1,1,bias=False)
        self.conv1_v1 = nn.Conv2d(self.exp, planes,1,1,0,bias=False, groups=inplanes)
        self.conv3_v1 = nn.Conv2d(planes, planes,3,1,1, bias=False)
        self.res = res
        if res:
            self.convres = nn.Conv2d(self.exp, planes,1,1,0,bias=False, groups=inplanes)
        self.f1 = nn.ELU() #(inplace=True) 
        self.bn1 = nn.BatchNorm2d(self.exp, planes)

    def forward(self, x):
        if self.res:
            residual = self.convres((self.DDC(x))) 
        x = self.f1(x)
        x = self.conv3_v1(self.f1(self.conv1_v1(self.DDC(x))))
        if self.res:
            x = x + residual
        return x 

class UpsamplingNearest2d(nn.Module): #(_UpsamplingBase):

    def __init__(self, size=None, scale_factor=None):
        super(UpsamplingNearest2d, self).__init__()
        self.scale_factor=scale_factor
        self.size=size

        if self.scale_factor is not None and not isinstance(scale_factor, Integral):
            raise ValueError('scale_factor must be a single Integer value for nearest neighbor sampling')

    def forward(self, input):
        assert input.dim() == 4

        if self.scale_factor is None:
            if (self.size[0] % input.size(2) != 0 or
                    self.size[1] % input.size(3) != 0):
                raise RuntimeError("output size specified in UpsamplingNearest "
                                   "({}) has to be divisible by the input size, but got: "
                                   "{}".format('x'.join(map(str, self.size)),
                                               'x'.join(map(str, input.size()))))
            self.scale_factor = self.size[0] // input.size(2)
            if self.scale_factor != self.size[1] // input.size(3):
                raise RuntimeError("input aspect ratio doesn't match the "
                                   "output ratio")

#        output = input.new() # not entirely sure what this is supposed to do..
        print "input:\n{}".format(input)
        output = input#.data * 0.0# works??
        backend = type2backend[type(input.data)]
        self.save_for_backward(input)
        backend.SpatialUpSamplingNearest_updateOutput(
            backend.library_state,
            input,
            output,
            self.scale_factor
        )
        print "OUTPUT:\n{}".format(output)
        return output

    def backward(self, grad_output):
        assert grad_output.dim() == 4

        input, = self.saved_tensors
        grad_input = grad_output.new()
        backend = type2backend[type(input)]
        backend.SpatialUpSamplingNearest_updateGradInput(
            backend.library_state,
            input,
            grad_output,
            grad_input,
            self.scale_factor
        )
        return grad_input

# Source: http://pytorch.org/docs/master/_modules/torch/nn/functional.html#upsample_nearest
#def upsample(input, size=None, scale_factor=None, mode='nearest'):
#    """Upsamples the input to either the given :attr:`size` or the given
#    :attr:`scale_factor`
#
#    The algorithm used for upsampling is determined by :attr:`mode`.
#
#    """
#    if input.dim() == 4 and mode == 'nearest':
#        return _functions.thnn.UpsamplingNearest2d(_pair(size), scale_factor)(input)
#    elif input.dim() == 5 and mode == 'nearest':
#        return _functions.thnn.UpsamplingNearest3d(_triple(size), scale_factor)(input)
#    elif input.dim() == 4 and mode == 'bilinear':
#        return _functions.thnn.UpsamplingBilinear2d(_pair(size), scale_factor)(input)
#    elif input.dim() == 4 and mode == 'trilinear':
#        raise NotImplementedError("Got 4D input, but trilinear mode needs 5D input")
#    elif input.dim() == 5 and mode == 'bilinear':
#        raise NotImplementedError("Got 5D input, but bilinear mode needs 4D input")
#    elif input.dim() == 5 and mode == 'trilinear':
#            return _functions.thnn.UpsamplingTrilinear3d(_triple(size), scale_factor)(input)
#    else:
#        raise NotImplementedError("Input Error: Only 4D and 5D input Tensors supported"
#                                  " (got {}D) for the modes: nearest | bilinear | trilinear"
#                                  " (got {})".format(input.dim(), mode))


class GlimpseSensor(nn.Module):
    def __init__(self, insize=896, outsize=224):
        super(GlimpseSensor, self).__init__() 
        if insize % outsize != 0:
            raise(RuntimeError("Glimpse ratio not even, aborting"))
        self.full = nn.Conv2d(1,1,kernel_size=7,stride=4,padding=3, bias=False)
        self.mid = nn.Conv2d(1,1,kernel_size=5,stride=2,padding=2, bias=False) 
        self.limiths = insize - outsize - 1
        self.limitls = outsize-1
        self.delta= outsize/2
        self.limitlm = self.limitls+self.delta
        self.limithm = self.limiths-self.delta
        self.medl = outsize*2
        self.focl = outsize
         
    def forward(self, input, x, y):
        xs = min(max(self.limitls,x), self.limiths)
        ys = min(max(self.limitls,y), self.limiths)
        xm = min(max(x-self.delta, self.limitlm), self.limithm) 
        ym = min(max(y-self.delta, self.limitlm), self.limithm)
        coarse = self.full(input)
#        medin = input[:,:,xm:xm+self.medl,ym:ym+self.medl].contiguous()
        med = self.mid(input[:,:,xm:xm+self.medl,ym:ym+self.medl])
#        med = self.mid(medin)
        focal = input[:,:,xs:xs+self.focl,ys:ys+self.focl]
        return cat([coarse, med, focal], 1)

class GlimpseRegression(nn.Module):
    def __init__(self, planes, insize):
        super(GlimpseRegression, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(planes, 1, kernel_size=(7,4), stride=(1,3), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.f = nn.ReLU(inplace=True)

    def forward(self, x):
        x = Variable(x.data) #separate so that gradient isn't overly backpropogated
        x = self.bn1(self.conv1(x))
        x = self.f(self.bn2(self.conv2(x)))
        x = self.f(self.conv3(x))
        return x[:,0,0,0], x[:,0,0,1]



#class DenseRecLayer(nn.Module):

#    def __init__(self, c,Height, Width):
#        super(DenseRecLayer, self).__init__()
#        self.in_H = Height
#        self.in_W = Width
#        self.downsize = 4
#        self.H = Height // self.downsize
#        self.W = Width //self.downsize
#        self.C = c
#        self.B = 1 # allow larger batches
#        rnns = []
#        self.downsample = nn.AvgPool2d(self.downsize, self.downsize, padding=0)
#        for i in range(self.H,* self.W):
#            rnns.append(nn.RNN(c, 2*c, 1))
#        self.rnns = nn.Sequential(*rnns)
#        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.downsize)
#        self.convdown = nn.Conv2d(2*c, c, kernel_size=1, bias=False)
#    def recur(self, h, x):
#        x_ = x.permute(0, 3, 1, 2).contiguous()
#        v = x_.view(self.B, self.H*self.W, self.C)
#        y_t = torch.Tensor(self.B, self.H*self.W, self.C)
#        h_t = torch.Tensor(self.B, self.H*self.W, self.C)
#        for i in range(v.size()[1]):
#            y_t[0, i, :], h_t[0,i,:] = self.rnns[i](v[0, i,:], h[0,i,:])
#        y_t = y_t.permute(0, 2, 1).contiguous()
#        h_t = h_t.permute(0,2,1).contiguous()
#        y_t = y_t.view(self.B, self.C, self.H, self.W)
#        h_t = h_t.view(self.B, self.C, self.H, self.W)
#        return y_t, h_t 
#        #x = x.view(x.size(0), -1)
#    def forward(self, h, x):
#        residual = x
#
#        out = self.downsample(x)
#        out, h_t = self.recur(x, h)
#        out = self.upsample(out)
#        out = torch.cat([residual, out], 1)
#        out = self.convdown(out)
#        return out, h





# Data Display Layers

class THNNLSM(nn.Module):
    def __init__(self, temp=1):
        super(THNNLSM, self).__init()       
        self.f = F.log_softmax
        self.temp = float(temp)
    def forward(self, input):
        return self.f(input/self.temp)



class EntropicFusion(nn.Module):
    def __init__(self, coarsech, semch):
        super(EntropicFusion, self).__init__()
        self.coarsech = coarsech
        self.semch = semch
        self.up = DUCSmooth(semch, coarsech, 2,3,False,'ID', False)

        self.convcoarse = nn.Sequential(
                            nn.BatchNorm2d(coarsech),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(coarsech, coarsech, 3,1,1,bias=False))
#        self.fuse = nn.Sequential(
#                            nn.Conv2d(coarsech*2, coarsech, 3,1,1,bias=False),
#                            nn.BatchNorm2d(coarsech),
#                            nn.ReLU())
        self.fuse_ = nn.Conv2d(coarsech, coarsech, 3,1,1,bias=False)

    def forward(self, coarse, sem, entr, gate=True): 
        coarse = self.convcoarse(coarse)
        if gate: #If False, deal with control scenari
            norm = entr.max()
            entr_ = entr / norm.expand_as(entr)
#            entr_ = (1.0-entr)
            entr = entr_.expand_as(coarse)
            # Expand to correct size
            #for i in range(self.coarsech-1):
            #    entr = cat([entr, entr_],1)
            coarse = (entr) * coarse 
        sem = self.up(sem)
        merged = sem + coarse
        merged = self.fuse_(merged)

#        merged = self.fuse(cat([coarse,sem],1))

        return merged
        
class EntropicGate(nn.Module):
    def __init__(self, inplanes, planes, r,kernel_size=1):
        super(EntropicGate, self).__init__()
        self.r = r
        self.gatesize = r//2
        self.up = DUCSmooth(inplanes, planes,r,kernel_size, False, 'ID', False)
        self.down = nn.AvgPool2d(self.gatesize+2,self.gatesize,1,count_include_pad=False)

    def forward(self, input):
        guess = self.up(input) #Return for a loss (prediction)
        entr = 1.0 - GetEntropy(guess) #Return for painting (depiction)
        entrdown = self.down(entr) # Return for gating (fiction?) 
        return guess, entr, entrdown



def GetEntropy(input):
        '''Expects a BxCxHxW tensor'''
#        x = input.detach()
        x = input
        lsm = F.log_softmax(x)
        sm = F.softmax(x)
        x = lsm * sm # easy -> 0; hard -> -inf
        x = x.sum(1)
        x = torch.exp(x) #easy -> 1; hard -> 0
        return x
#        return x.detach()


def SaveFuzzyEntropy(im, entropy, modelname, layer, index, entr2 = None, entr3=None):
    root = '/data/hinthorn/workspace_hinthorn/exp/pytorch'
    savename = os.path.join(root, 'imgs','entropy_'+modelname +'_'+ layer+'_frame_'+ str(index)+ '.png')
#    print "Saving: {}".format(savename)
    c,h,w = im.size()
    
    c_,h_,w_ = entropy.size()

    entropy =    entropy.expand(1,1,h_,w_)
    if h_ < h:
        entropy = F.upsample_nearest(entropy, size=(h,w)) #,mode='nearest')
    entropy= entropy.data.cpu().squeeze().numpy() 
    entropy = (entropy / np.max(entropy)) *250
    im = im.data.cpu().squeeze().numpy()
    im = (im - np.min(im))
    im = (im / np.max(im)) * 200 


        
    temp = np.zeros([h, w, 3], dtype=np.float32)
    for i in range(0,3):
        temp[:,:, i] = im
    im = temp
    alpha = 0.3

    if entr2:
        c_,h_,w_ = entr2.size()
        entr2 =    entr2.expand(1,1,h_,w_)
        if h_ < h:
            entr2 = F.upsample_nearest(entr2, size=(h,w)) 
        entr2= entr2.data.cpu().squeeze().numpy() 
#        temp = np.max(entr2)
#        if temp > big:
#            big = temp
        entr2 = (entr2 / np.max(entr2)) *250
        im[:,:, 0] = im[:,:,0] * alpha + (1-alpha) * entr2[:,:] #Blue

    if entr3:
        c_,h_,w_ = entr3.size()
        entr3 =    entr3.expand(1,1,h_,w_)
        if h_ < h:
            entr3 = F.upsample_nearest(entr3, size=(h,w)) 
        entr3= entr3.data.cpu().squeeze().numpy() 
        entr3 = (entr3 / np.max(entr3)) *250
        im[:,:, 1] = im[:,:,1] * alpha + (1-alpha) * entr3[:,:] #Green

#        temp = np.max(entr3)
#        if temp > big:
#            big = temp
#    entropy =entropy/big*250


    im[:,:, 2] = im[:,:,2] * alpha + (1-alpha) * entropy[:,:] #red

#    if type(entr2)!=None:
#        entr = entr2/big*250
#        im[:,:, 0] = im[:,:,0] *alpha + (1-alpha) * entr2[:,:] #Blue
#    if type(entr3)!=None:
#        entr3 = entr3/big*250
#        im[:,:, 1] = im[:,:,1] *alpha + (1-alpha) * entr3[:,:] #green

    im = np.array(im, dtype=np.uint8) 
    cv2.imwrite(savename, im)





class im2dist(nn.Module):
    def __init__(self, kern):
        self.lam = Variable(torch.zeros(1)) 
        self.k = k
        self.im2col = im2col(kern, kern,1,1,1,1,1,1)

    def forward(self, input):
        print "Lambda: {}".format(self.lam.data) 
        b,c,h,w = input.size()
        diff = self.im2col(input)
        b_,c_,h_,w_,k = diff.size()
        for i in range(1, k):
            print "AAAAA"
        out = out * self.lamb.expand_as(out)
        out = torch.exp(out)
        return out 


''' im2col(input, kH, kW, dH, dW, padH, padW, sH, sW):'''
class SegAwareConv(nn.Module):
    def __init__(self,c, c_out, kern, lam): 
        self.im2col = im2col(kern, kern,1,1,1,1,1,1)
        self.im2dist = im2dist(kern, lam)
        self.weights = nn.Linear(c, c_out) 

# Find a way to store params as a single matrix
    def forward(self, image, embeddings):
        b,c,h,w = image.size()
        embeddings = self.im2dist(embeddings)
        im_unroll = self.im2col(image)
        out = im_unroll*embeddings
        out = out.mm(self.weights)
        out = out.view(b,c,h,w)
        return out



