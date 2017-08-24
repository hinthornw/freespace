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

__all__ = ['EntropyNet']




index = 0

class EntropyNetDUC(nn.Module):
    '''Model 2 of Recursive search. We test a convolutional Elman network tacked on the end of our original CNN architecture. Test a two-path ending so that frame 1 of n for a given sequence bypasses the hidden architecture but feeds the input for the rnn later.'''
    

    def __init__(self, block, layers, dilations={0:None, 1:None, 2:None, 3:None, 4:None},num_labels=2, mode='iid', fuse=False, seqLength=1):
        super(EntropyNetDUC, self).__init__()
        self.inplanes = 64
        self.fuse= fuse

        self.mode = mode
        self.num_labels=num_labels 
        self.seqLength = seqLength
        
        self.convin = nn.Sequential( #Does adding nonlinearity help? -WILL
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                                   bias=False),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        
        strides = [2,2,2,1] # 224: 112->56->28->
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=strides[0], dil=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dil=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dil=dilations[2])

        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dil=dilations[3])

        self.gate4 = EntropicGate(512, 2, 16, kernel_size=3)
        self.entrfuse2 = EntropicFusion(128, 512)   
        self.gate2 = EntropicGate(128, 2, 8, kernel_size=3)
        self.entrfuse1 = EntropicFusion(64, 128)
        self.gate1 = EntropicGate(64, 2, 4, kernel_size=3)
        self.entrfuse0 = EntropicFusion(64, 64)
        self.upfinal = DUCSmooth(64, 4, 2, 3, False, 'ELU', False)

        self.convout= nn.Sequential(
                            nn.Conv2d(4,4,3,1,1,bias=False),
                            nn.BatchNorm2d(4),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(4,2,3,1,1, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(2,2,3,1,1,bias=True))



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


#    def init_hidden(self):
#        b, c, h, w = self.b, self.rnn_hidden_size, self.rnn_h, self.rnn_w
#        return Variable(zeros(b, c, h, w))


    def forward(self, x):
        global index
    
        b = x.size(0) 
        if b % self.seqLength != 0:
            raise(RuntimeError('Flexible Seq-batch ratios not yet implemented'))
        
        outstem = self.convin(x)
        out1 = self.layer1(outstem)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        
        guess4, entr4, entrdown4 = self.gate4(out4) 
        fused2 = self.entrfuse2(out2, out4, entrdown4, gate=self.fuse)
        guess2, entr2, entrdown2 = self.gate2(fused2)
        fused1 = self.entrfuse1(out1, fused2, entrdown2, gate=self.fuse)
        guess1, entr1, entrdown1 = self.gate1(fused1)
        fused0 = self.entrfuse0(outstem, fused1,entrdown1, gate=self.fuse)
        out = self.upfinal(fused0)
        outsmooth = self.convout(out)


        return outsmooth, guess1, guess2, guess4 



def EntropyNet(num_labels=2, seqLength=1,mode='iid',fuse=False, **kwargs):
    """Constructs a dense kalman model. CONV-RNN on end with an initializer for the first frame in a sequence.

    Modifications by Will for Semantic Segmentation - added HDC in blocks 2 & 3 of values(block 0 & 1 left untouched):
    2: [1, 2, 3, 5, 9, 17],
    3: [5, 2, 1] #Based on experiments with TidalWave
    """
    #dil= {0: [1,1,1], 1: [1,1,1,1], 2:[1,2,3,5,9,17],3:[5,2,1]} #Based on experiments with TidalWave
    dil= {0: [(1,1),(1,1)], 1: [(1,1),(1,1),], 2:[(1,1),(1,1)],3:[(1,1),(2,2),(3,3),(1,1)], 4:[(1,1),(1,1)]} #Based on HDC
    model = EntropyNetDUC(BasicBlockx, [2,2,2,4,2], dil, num_labels, mode, fuse)
    return model
