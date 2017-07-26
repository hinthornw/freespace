import sys
import torch.nn as nn
sys.path.append('./ark')
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch import zeros, cat
import time
from layers import *
from collections import OrderedDict

__all__ = ['Kalman']





class KalmanStack(nn.Module):
    '''Model 2 of Recursive search. We test a convolutional Elman network tacked on the end of our original CNN architecture. Test a two-path ending so that frame 1 of n for a given sequence bypasses the hidden architecture but feeds the input for the rnn later.'''
    
    def __init__(self, block, layers, dilations={0:None, 1:None, 2:None, 3:None, 4:None}, recType=(ConvElman, ConvElman), num_labels=2, mode='iid'):
        super(KalmanStack, self).__init__()
        self.inplanes = 64


        self.num_labels=num_labels 
        
        self.convin = nn.Sequential( #Does adding nonlinearity help? -WILL
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                                   bias=False),
            #nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.smoothstem = nn.Sequential(
                            nn.Conv2d(64,64,3,1,1,bias=False),
                            nn.ELU())
        
        #Make 4 blocks, each doubling the channel size, the last three halving FM H&W
        strides = [2,2,2,2,2] # 224: 112->56->28->14->7
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=strides[0], dil=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dil=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dil=dilations[2])
        self.rnn_hidden_size = 512 
        self.rec = ConvElman(256, self.rnn_hidden_size,3)
        self.rnn_h = 28
        self.rnn_w = 28
        self.b = 1
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dil=dilations[3])
        self.layer5 = self._make_layer(block, 512, layers[4], stride=strides[4], dil=dilations[4])

        self.fiveup = DUCSmooth(512, 128, strides[4]*strides[3]*strides[2],1,False)
        self.fourup = DUCSmooth(512, 128, strides[3]*strides[2],1,False)
        self.level2Smooth = nn.Sequential(
                            nn.Conv2d(128,64,1,1,0,bias=False),
                            nn.Conv2d(64,64,3,1,1,groups=16,bias=False),
                            nn.ELU(inplace=True),
                            nn.Conv2d(64,128,1,1,0,bias=False))
        self.threeup = DUCSmooth(256,32, strides[2]*strides[1],1,False)#mergew/lv1

        self.twoup = DUCSmooth(128, 32, strides[1]*strides[0]*2,3,False)#mergew/out
        self.twogate = nn.Sequential(
                     DUCSmooth(128, 64, strides[1]*strides[0],3, True),
                     nn.Conv2d(64,64,3,1,1,bias=False),
                     nn.Sigmoid()) #gate stem

        self.oneup = DUCSmooth(64, 32, strides[0]*2,1,True)#mergew/out

        self.lv1smooth1 = nn.Sequential(
                                nn.Conv2d(64,32,3,1,1,bias=False),
                                nn.ELU(inplace=True))

        self.lvl1smooth2= nn.Sequential(
                            nn.Conv2d(64,32,1,1,0,bias=False),
                            nn.Conv2d(32,32,3,1,1,groups=8,bias=False),
                            nn.ELU(inplace=True),
                            nn.Conv2d(32,64,1,1,0,bias=False))

        self.stemup = nn.Sequential(
                        nn.ConvTranspose2d(64, 32, 3, 2, 1,output_padding=1, bias=False),
                        nn.ELU(inplace=True),
                        nn.Conv2d(32,16,1,1,0,bias=False),
                        nn.Conv2d(16,16,3,1,1,groups=4,bias=False),
                        nn.ELU(inplace=True),
                        nn.Conv2d(16,32,1,1,0,bias=True)) 
        self.predict = nn.Sequential(OrderedDict([
                  ( 'comp', nn.Conv2d(96, 32, 1,1,0,bias=False)),
                  ( 'act1', nn.ELU(inplace=True)),
                  ( 'conv2', nn.Conv2d(32,32, 3,1,1,groups=8, bias=False)),
                  ( 'act' , nn.ELU(inplace=True)),
                  ( 'out'+str(self.num_labels),
                    nn.Conv2d(32, self.num_labels,3,1,1,bias=True))])) 
        self.f = nn.ELU(inplace=True)
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
        return Variable(zeros(b, c, h, w))

    def forward(self, x, h, i):
        outstem = self.convin(x)
        out1 = self.layer1(outstem)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        y, h = self.rec(out3, h)
        if i > 0:
            out3 = out3 + y
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out5 = self.fiveup(out5)
        out4 = self.fourup(out4)
        out2 = self.level2Smooth(out2 + out5 + out4)
        twogate = self.twogate(out2)
        out2 = self.twoup(out2)
        out3 = self.threeup(out3)
        out1 = self.lv1smooth1(out1)
        out1 = self.lvl1smooth2(cat([out1, out3],1))
        out1 = self.oneup(out1)
        outstem = self.f(self.smoothstem(outstem) * twogate)
        outstem = self.stemup(outstem)
        out = self.predict(cat([outstem, out2, out1], 1))
        return out, h


def Kalman(num_labels=2, insize=448,mode='iid', **kwargs):
    """Constructs a dense kalman model. CONV-RNN on end with an initializer for the first frame in a sequence.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    Modifications by Will for Semantic Segmentation - added HDC in blocks 2 & 3 of values(block 0 & 1 left untouched):
    2: [1, 2, 3, 5, 9, 17],
    3: [5, 2, 1] #Based on experiments with TidalWave
    """
    #dil= {0: [1,1,1], 1: [1,1,1,1], 2:[1,2,3,5,9,17],3:[5,2,1]} #Based on experiments with TidalWave
    dil= {0: [(1,1),(1,1)], 1: [(1,1),(1,1),], 2:[(1,1),(1,1)],3:[(1,1),(1,1)], 4:[(1,1),(1,1)]} #Based on HDC
    if mode == 'rec':
        model = KalmanStack(BasicBlockx, [2,2,2,2,2], dil, (ConvElman, ConvElman), num_labels, mode)
    else:
        model = KalmanStack(BasicBlockx, [2,2,2,2,2], dil, (ConvElman, ConvElman), num_labels)
    return model
