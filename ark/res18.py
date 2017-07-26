import sys
import torch.nn as nn
sys.path.append('./ark')
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import time
from layers import *



__all__ = ['ResNeXt18']




class ResNext(nn.Module):

    def __init__(self, block, layers, dilations={1:None, 2:None, 3:None, 4:None}, stem='reg', insize=448):
        super(ResNext, self).__init__()
        self.inplanes = 64
        outstride=8
        if stem == 'HD':
            outstride = 64
#        layers_outsize=256
        label_types=2
        self.stem = stem
        if stem == 'reg' or stem == 'regdense':
            self.convin = nn.Sequential( #Does adding nonlinearity help? -WILL
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                                       bias=False),
                #nn.BatchNorm2d(16),
                nn.ELU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ELU(inplace=True),
                #nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64))
        elif stem == 'HD':
            self.convHD = HDInput(1, 64, insize) 
        else:
            raise(RuntimeError("Stem {} not legal.".format(stem)))
#        outsize = [128, 256, 512, 512]
        self.outsize = None
        if stem == 'regdense':
            self.outsize = [128, 256, 512, 512]
            self.f = nn.ELU(inplace=True) #nn.ReLU(inplace=True)
            self.layer1_v1 = self._make_layer(block, self.outsize[0], layers[0], stride=2, dil=dilations[0])
            self.layer2_v1= self._make_layer(block, self.outsize[1], layers[1], stride=2, dil=dilations[1])
#            self.denser = DenserFusion(self.outsize[1], self.outsize[1])
            self.layer3_v1 = self._make_layer(block, self.outsize[2], layers[2], stride=2, dil=dilations[2]) 
            #self.atrous = DenseFusion(self.outsize[2], self.outsize[2]/2)
            self.layer4_v1 = self._make_layer(block, self.outsize[3], layers[3], stride=2, dil=dilations[3])
            self.smoother_v1 = block(self.inplanes, self.inplanes, stride=1, dil=[(1,1),(1,1)],downsample=None) 
        elif stem == 'reg'or stem == 'HD':
            self.outsize = [64, 128, 256, 256]
            self.f = nn.ELU(inplace=True) #nn.ReLU(inplace=True)
            self.layer1= self._make_layer(block, self.outsize[0], layers[0], stride=2, dil=dilations[0])
            self.layer2_= self._make_layer(block, self.outsize[1], layers[1], stride=2, dil=dilations[1])
            self.layer3_= self._make_layer(block, self.outsize[2], layers[2], stride=1, dil=dilations[2]) 
            self.layer4_= self._make_layer(block, self.outsize[3], layers[3], stride=1, dil=dilations[3])
            self.smoother= block(self.inplanes, self.inplanes, stride=1, dil=[(1,1),(1,1)],downsample=None) 



        
        if stem == 'HD':
            self.DUCHD = DUC(self.outsize[3], label_types, outstride)
        elif stem == 'reg':
            self.DUC = DUC(self.outsize[3], label_types, outstride)
        else:
            self.DUC_v1 = DUC(self.outsize[3], label_types, outstride)
            self.DUC_l2_v1 = DUC(self.outsize[1], label_types, outstride)
            self.DUC_l1_v1= DUC(self.outsize[0], label_types, outstride/2)
            self.DUC_stem = DUC(64, label_types, outstride/4)
            self.fusedout = DenseFusion(label_types*4, label_types*4)   
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,dil=[0]):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dil[0], downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dil=dil[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.stem == 'reg' or self.stem == 'regdense':
            x = self.convin(x)
            x = self.f(x)
        elif self.stem == 'HD':
            x = self.convHD(x)
        else:
            raise(RuntimeError("Illegal stem")) 
        if self.stem == 'regdense':
            res = x
            x1 = self.layer1_v1(x)
            x2 = self.layer2_v1(x1)
            x2 = self.denser(x2) # I think it's OK to be doing this?
            x = self.layer3_v1(x2)
            x = self.smoother_v1(x)
            x = self.DUC_v1(x)
            res = self.DUC_stem(res)
            x1 = self.DUC_l1_v1(x1)
            x2 = self.DUC_l2_v1(x2)
            x = cat([res,x1,x2,x],1)



        else:
            x = self.layer1(x)
            x = self.layer2_(x)
            x = self.layer3_(x)
            x = self.layer4_(x)
            x = self.smoother(x)
#            print 'after smoother: {}'.format(x.size())
            if self.stem == 'reg':
                x = self.DUC(x)
            elif self.stem == 'HD':
                x = self.DUCHD(x)
#            print('outsize:\t{}'.format(x.size()))
        return x 

def ResNeXt18(pretrained=False, stem='reg', insize=448, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    dil= {0: [(1,1), (1,1)], 1: [(1,2),(1,1)], 2:[(2,3),(2,3)],3:[(5,9), (5,9)]} #Based on HDC
    model = ResNext(BasicBlockx, [2, 2, 2, 2], dilations=dil, stem=stem, insize=insize)
        
    return model
