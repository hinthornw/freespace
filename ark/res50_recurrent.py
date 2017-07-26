import torch.nn as nn
import sys
sys.path.append('./ark')
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch import ones, cat
import time
from layers import *
#__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           #'resnet152']
__all__ = ['resrecElm50', 'resrecDualElm50', 'resrecGRU50', 'resrecDualGRU50']

model_urls = {
#    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnetDUC34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnetDUC50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


#############################################################################
# Above is the base res50 model. Below, we explore recurrent architectures #
#                                                                          #
#                                                                          #
############################################################################

class ResRec50(nn.Module):
    '''Model 1 of Recursive search. We test a convolutional Elman network tacked on the end of our original CNN architecture.
    '''
    def __init__(self, block, layers, dilations={1:None, 2:None, 3:None, 4:None}, recType=ConvElman):
        super(ResRec50, self).__init__()
        self.inplanes = 64
        layers_outsize = 2048 #512
        self.rnn_h = 56 #60
        self.rnn_w =56 # 60
        self.b = 1
        self.rnn_hidden_size = layers_outsize // 2
        label_types = 2 #(i.e. 0,1)
        outstride = 8
        self.conv1_1D = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
#        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downstem = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        #Make 4 blocks, each doubling the channel size, the last three halving FM H&W
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, dil=dilations[1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dil=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dil=dilations[2])
        self.layer4 = self._make_layer(block, layers_outsize//4, layers[3], stride=1, dil=dilations[3])
        self.rnn = RecLayer(layers_outsize, recType)
        self.DUC = DUC(layers_outsize, label_types, outstride)
        for m in self.modules(): # initialize weights
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels # k**2 * C_out
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dil=[1]):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )#1d conv and BN

        layers = []
        layers.append(block(self.inplanes, planes, stride, dil[0], downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dil=dil[i]))

        return nn.Sequential(*layers)
    def init_hidden(self):
        b, c, h, w = self.b, self.rnn_hidden_size, self.rnn_h, self.rnn_w
        return Variable(ones(b, c, h, w))
    def forward(self, x, h, i):
        out= self.conv1_1D(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.downstem(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out, h_t = self.rnn(out, h)
        out = self.DUC(out)
        return out, h_t


def resrecElm50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    Modifications by Will for Semantic Segmentation - added HDC in blocks 2 & 3 of values(block 0 & 1 left untouched):
    2: [1, 2, 3, 5, 9, 17],
    3: [5, 2, 1] #Based on experiments with TidalWave
    """
    #dil= {0: [1,1,1], 1: [1,1,1,1], 2:[1,2,3,5,9,17],3:[5,2,1]} #Based on experiments with TidalWave
    dil= {0: [1,1,1], 1: [1,1,1,1], 2:[2,2,3,3,5,5],3:[1,1,1]} #Based on HDC
    model = ResRec50(Bottleneck, [3, 4, 6, 3], dil, ConvElman)
    if pretrained:
        model_dict = model.state_dict()
        reference =  model_zoo.load_url(model_urls['resnetDUC50'])
        pretrained_dict = {k:v for k, v in reference.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def resrecGRU50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    Modifications by Will for Semantic Segmentation - added HDC in blocks 2 & 3 of values(block 0 & 1 left untouched):
    2: [1, 2, 3, 5, 9, 17],
    3: [5, 2, 1] #Based on experiments with TidalWave
    """
    #dil= {0: [1,1,1], 1: [1,1,1,1], 2:[1,2,3,5,9,17],3:[5,2,1]} #Based on experiments with TidalWave
    dil= {0: [1,1,1], 1: [1,1,1,1], 2:[2,2,3,3,5,5],3:[1,1,1]} #Based on HDC
    model = ResRec50(Bottleneck, [3, 4, 6, 3], dil, ConvGRU)
    return model



class ResRecDual50(nn.Module):
    '''Model 2 of Recursive search. We test a convolutional Elman network tacked on the end of our original CNN architecture. Test a two-path ending so that frame 1 of n for a given sequence bypasses the hidden architecture but feeds the input for the rnn later.'''
    
    def __init__(self, block, layers, dilations={1:None, 2:None, 3:None, 4:None}, recType=(ConvElman, ConvElman)):
        super(ResRecDual50, self).__init__()
        self.inplanes = 64
        layers_outsize = 2048 #512
        self.rnn_h = 56 #60
        self.rnn_w =56 # 60
        self.b = 1
        self.rnn_hidden_size = layers_outsize // 2
        label_types = 2 #(i.e. 0,1)
        outstride = 8
        self.conv1_1D = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.downstem = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        #Make 4 blocks, each doubling the channel size, the last three halving FM H&W
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, dil=dilations[1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dil=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dil=dilations[2])
        self.layer4 = self._make_layer(block, layers_outsize//4, layers[3], stride=1, dil=dilations[3])
        self.rnn = RecLayer(layers_outsize, recType[0])
        self.rnninit = RecLayer(layers_outsize, recType[1])
        self.DUC = DUC(layers_outsize, label_types, outstride)
        self.paramout = nn.Sequential(
                nn.Conv2d(label_types, label_types*2, kernel_size=(10,3),stride=(8,1),
                    padding=(1,1), bias=False),
                nn.ELU(),
                nn.Conv2d(label_types*2, label_types*2, kernel_size=(10, 3), stride=(8,1), padding=(1,1), bias=False),
                nn.ELU(),
                nn.Conv2d(label_types*2, 2, kernel_size=(7,1),
                    stride=1, padding=0, bias=False))
        self.out3 = nn.Sequential(
                nn.Conv2d(layers_outsize//2, layers_outsize, kernel_size=1, padding=0),
                nn.Conv2d(layers_outsize, layers_outsize, kernel_size=3, padding=1), 
                nn.ELU(),
                nn.Conv2d(layers_outsize, layers_outsize, kernel_size=3, padding=1),
    #            nn.BatchNorm2d(layers_outsize),
                nn.ELU(),
                DUC(layers_outsize, label_types, outstride))
        self.out2 = nn.Sequential(
                nn.Conv2d(layers_outsize//2, layers_outsize, kernel_size=1, padding=0),
                nn.Conv2d(layers_outsize, layers_outsize, kernel_size=3, padding=1), 
                nn.ELU(),
                nn.Conv2d(layers_outsize, layers_outsize, kernel_size=3, padding=1),
    #            nn.BatchNorm2d(layers_outsize),
                nn.ELU(),
                DUC(layers_outsize, label_types, outstride))
        for m in self.modules(): # initialize weights
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels # k**2 * C_out
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dil=[1]):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )#1d conv and BN

        layers = []
        layers.append(block(self.inplanes, planes, stride, dil[0], downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dil=dil[i]))

        return nn.Sequential(*layers)
    def init_hidden(self):
        b, c, h, w = self.b, self.rnn_hidden_size, self.rnn_h, self.rnn_w
        return Variable(ones(b, c, h, w))
    def forward(self, x, h, i):
        out= self.conv1_1D(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.downstem(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        h_t = None
        if i > 0:
            out, h_t = self.rnn(out, h)
        else:
            out, h_t = self.rnninit(out, h)
        
        if self.training:
            out2 = self.out2(h_t) 
            out3 = self.out3(h_t)
            out = self.DUC(out)
            outp = self.paramout(out)
            outp = outp.transpose(1,2).contiguous()
#            print "out3, outp:\t{}, {}".format(out3.size(), outp.size())
            return out, out2, out3, outp, h_t
        else:
            out = self.DUC(out)
            return out, h_t

def resrecDualElm50(pretrained=False, **kwargs):
    """Constructs a ResNetDual-50 model. CONV-RNN on end with an initializer for the first frame in a sequence.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    Modifications by Will for Semantic Segmentation - added HDC in blocks 2 & 3 of values(block 0 & 1 left untouched):
    2: [1, 2, 3, 5, 9, 17],
    3: [5, 2, 1] #Based on experiments with TidalWave
    """
    #dil= {0: [1,1,1], 1: [1,1,1,1], 2:[1,2,3,5,9,17],3:[5,2,1]} #Based on experiments with TidalWave
    dil= {0: [1,1,1], 1: [1,1,1,1], 2:[2,2,3,3,5,5],3:[1,1,1]} #Based on HDC
    model = ResRecDual50(Bottleneck, [3, 4, 6, 3], dil, (ConvElman, ConvElman))
    return model
def resrecDualGRU50(pretrained=False, **kwargs):
    """Constructs a ResNetDual-50 model. CONV-RNN on end with an initializer for the first frame in a sequence.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    Modifications by Will for Semantic Segmentation - added HDC in blocks 2 & 3 of values(block 0 & 1 left untouched):
    2: [1, 2, 3, 5, 9, 17],
    3: [5, 2, 1] #Based on experiments with TidalWave
    """
    #dil= {0: [1,1,1], 1: [1,1,1,1], 2:[1,2,3,5,9,17],3:[5,2,1]} #Based on experiments with TidalWave
    dil= {0: [1,1,1], 1: [1,1,1,1], 2:[2,2,3,3,5,5],3:[1,1,1]} #Based on HDC
    model = ResRecDual50(Bottleneck, [3, 4, 6, 3], dil, (ConvGRU, ConvElman))
    return model

