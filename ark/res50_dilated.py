import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch import ones, cat
import time
from layers import *
#__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           #'resnet152']
__all__ = ['resnetDUC50']

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

class ResNet50(nn.Module):
    '''Model 1 of Recursive search. We test a convolutional Elman network tacked on the end of our original CNN architecture.
    '''
    def __init__(self, block, layers, dilations={1:None, 2:None, 3:None, 4:None}, recType=ConvElman):
        super(ResNet50, self).__init__()
        self.inplanes = 64
        layers_outsize = 2048 #512
        self.rnn_h = 56 #60
        self.rnn_w =56 # 60
        self.b = 1
        self.rnn_hidden_size = layers_outsize // 2
        label_types = 2 #(i.e. 0,1)
        outstride = 8
        
        #self.retina = nn.Conv2d(1,1,kernel_size=27, stride=1, padding=13, bias=False)

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
        self.smoother = nn.Sequential(
                        nn.Conv2d(layers_outsize, layers_outsize, kernel_size=3, padding=1),
                        nn.ELU(),
                        nn.Conv2d(layers_outsize, layers_outsize, kernel_size=3, padding=1),
                        nn.BatchNorm2d(layers_outsize),
                        nn.ReLU()
                        ) 
        self.DUC = DUC(layers_outsize, label_types, outstride)
        #retina = True 
        def distance(x, y, sz=27):
            md = (sz-1)//2
            dist = 1.0 + ((x-md)**2 + (y-md)**2)
            return 1.0/dist
            
        for m in self.modules(): # initialize weights
            if type(m) == ResNet50:
                pass
            #elif retina:
            #    sz=27
            #    for i in range(sz):
            #        for j in range(sz):
            #            m.weight[0,0,i,j].data.fill_(distance(i,j))
            #    retina = False
            #    #print m.weight
            elif isinstance(m, nn.Conv2d):
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
    def forward(self, x):
        out = x
        #out = self.retina(out)
        out = self.conv1_1D(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.downstem(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.smoother(out)
        out = self.DUC(out)
        return out


def resnetDUC50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    Modifications by Will for Semantic Segmentation - added HDC in blocks 2 & 3 of values(block 0 & 1 left untouched):
    2: [1, 2, 3, 5, 9, 17],
    3: [5, 2, 1] #Based on experiments with TidalWave
    """
    #dil= {0: [1,1,1], 1: [1,1,1,1], 2:[1,2,3,5,9,17],3:[5,2,1]} #Based on experiments with TidalWave
    dil= {0: [1,1,1], 1: [1,2,3,5], 2:[1,2,3, 5, 7, 5],3:[3,2,1]} #Based on HDC
    model = ResNet50(Bottleneck, [3, 4, 6, 3], dil, ConvElman)
    return model

