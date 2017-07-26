import sys
import torch.nn as nn
sys.path.append('./ark')
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import time
from torch import FloatTensor, zeros, cat
from layers import *



__all__ = ['ResGlimpse18']

# Original Input Size 1200 x 1200 (squashed)
# Current input dimensions 1x448x448
# New Input Dimensions: 3x224x224
# So input (1200->224), (600->224), (224) 


class ResGlimpse(nn.Module):
    def __init__(self, block, layers, dilations={1:None, 2:None, 3:None, 4:None}, stem='reg', recType=ConvElman):
        super(ResGlimpse, self).__init__()
        self.inplanes = 64
        outstride=8
        if stem == 'HD':
            outstride = 16
        layers_outsize=256
        label_types=2
        self.input_size = 896
        self.glimpse_size = 224
        self.stem = stem
        self.rnn_h = 28 
        self.rnn_w = 28 
        self.b = 1
        self.rnn_hidden_size = layers_outsize * 2
        
        self.glimpse = GlimpseSensor(self.input_size, self.glimpse_size) 
        self.convin_ = nn.Sequential( #Does adding nonlinearity help? -WILL
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                                   bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64))
        
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2, dil=dilations[0])
        self.layer2_= self._make_layer(block, 128, layers[1], stride=2, dil=dilations[1])
        self.layer3_ = self._make_layer(block,256, layers[2], stride=1, dil=dilations[2]) 
        self.layer4_ = self._make_layer(block, 256, layers[3], stride=1, dil=dilations[3])
        self.smoother = block(256, 256, stride=1, dil=[(1,1),(1,1)],downsample=None) 
        self.rnn = RecLayerBasic(layers_outsize, recType)
        self.squash = GlimpseRegression(256, self.glimpse_size/outstride) 
        
        if stem == 'HD':
            self.DUCHD = DUC(layers_outsize, label_types, outstride)
        else:
            self.DUC = DUC(layers_outsize, label_types, outstride)

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

    def init_glimpse(self):
        start = self.input_size/2 -self.glimpse_size/2
        return Variable(FloatTensor(start)), Variable(FloatTensor(start)) 

    def init_hidden(self):
        b, c, h, w = self.b, self.rnn_hidden_size, self.rnn_h, self.rnn_w
        return Variable(zeros(b, c, h, w))
    def forward(self, x, y, z, h):
        # x is input, (y,z) are height, width coordinates of the center of the glimpse
        x = self.glimpse(x, y, z) 
    #    if self.stem == 'reg':
    #        x = self.convin(x)
    #        x = self.relu(x)
    #    elif self.stem == 'HD':
    #        x = self.convHD(x)
    #    else:
    #        raise(RuntimeError("Illegal stem"))
        x = self.relu(self.convin_(x)) 
        x = self.layer1(x)
        x = self.layer2_(x)
        x = self.layer3_(x)
        x = self.layer4_(x)
        x = self.smoother(x)
        x, h_t= self.rnn(x, h)
        y,z = self.squash(x)
        x = self.DUC(x)
        print "Y,Z\t|\t{}, {}".format(y.data, z.data)
        return x, y, z, h_t 

def ResGlimpse18(pretrained=False, stem='reg', **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    dil= {0: [(1,1), (1,1)], 1: [(1,2),(1,1)], 2:[(2,3),(2,3)],3:[(5,9), (5,9)]} #Based on HDC
    model = ResGlimpse(BasicBlockx, [2, 2, 2, 2], dilations=dil, stem=stem)
        
    return model
