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

__all__ = ['FCNDUCLONG']




index = 0

class FCNKalman(nn.Module):
    '''Model 2 of Recursive search. We test a convolutional Elman network tacked on the end of our original CNN architecture. Test a two-path ending so that frame 1 of n for a given sequence bypasses the hidden architecture but feeds the input for the rnn later.'''
    

    def __init__(self, block, layers, dilations={0:None, 1:None, 2:None, 3:None, 4:None},num_labels=2, mode='iid', seqLength=1):
        super(FCNKalman, self).__init__()
        self.inplanes = 64

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
#        self.up2 = DUCSmooth(128, 2, 8,3,False,'ID')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dil=dilations[2])
#        self.up3 = DUCSmooth(256, 2, 16,3,False,'ID')

        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dil=dilations[3])
        self.smooth = nn.Sequential(
                        nn.ELU(inplace=True),
                        nn.Conv2d(512,512,3,1,1,bias=False),
                        nn.ELU(inplace=True),
                        nn.Conv2d(512,512,3,1,1,bias=False))

        self.recmo = nn.Conv2d(512,512,3,1,1,bias=False)
        self.recfuse = nn.Conv2d(512,512,3,1,1,bias=False)
        self.reca = nn.Conv2d(512,512,3,1,1,bias=False)

#        self.rec1 = MultiLayerRecurrent(512) 
#        self.rec2 = nn.Conv2d(512, 512, 3,1,1,bias=False) 
        self.fourup1 = DUCSmooth(512, 128,4,3,False,'ID') 
        self.fourup2 = DUCSmooth(128, 8,4, 3, False, 'ID')
        self.outconv = nn.Conv2d(8,2,3,1,1)
#        self.fuse = nn.Conv2d(self.num_labels*2, self.num_labels,5,1,2)
        self.f = nn.ELU()

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




    def forward(self, x, prev, hi, acc):
        global index
    
        b = x.size(0) 
        if b % self.seqLength != 0:
            raise(RuntimeError('Flexible Seq-batch ratios not yet implemented'))
        
        outstem = self.convin(x)
        out1 = self.layer1(outstem)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out4 = self.smooth(out4)

#        if self.mode == 'kalman':
#            #Predictive Correlative Block
#            b,c,h,w = out4.size()
#            seq1 = []
#            if not hi:
#                hi = Variable(out4[0].data.expand(1,c,h,w)*0.0)
#            if not prev:
#                prev = out4[0].expand(1,c,h,w)
#
#            for i in range(b):
#                curr = out4[i].expand(1,c,h,w)
#                out, hi = self.rec1(curr, prev, hi) 
#                seq1.append(out)
#                prev = curr
#            out = torch.cat(seq1,0)
#        elif self.mode == 'iid':
#            out = self.rec2(out4)
        if self.mode == 'kalman':
            b,c,h,w = out4.size()
            seq = []
            if not hi:
                hi = Variable(out4.narrow(0,0,1).data*0.0)
            if not prev:
                prev = out4.narrow(0,0,1)
            for i in range(b):
                curr = out4.narrow(0,i,1)
                out = curr + self.recmo(curr - prev)
                prev = curr
                seq.append(out)
            out = torch.cat(seq, 0)

        elif self.mode == 'kalmana':
            b,c,h,w = out4.size()
            seq = []
            if not hi:
                hi = Variable(out4.narrow(0,0,1).data*0.0)
            if not prev:
                prev = out4.narrow(0,0,1)
                acc = out4.narrow(0,0,1) * 0.0
            for i in range(b):
                curr = out4.narrow(0,i,1)
                mo = self.recmo(curr - prev)
                acc = acc+self.reca(mo + acc)
#                kin = self.recfuse(acc)
                out = curr + mo + acc 
                out = self.recfuse(out)
                prev = curr
                seq.append(out)
            out = torch.cat(seq, 0)

        elif self.mode == 'iidT':
            out = out4
        elif self.mode == 'iid':
            out = self.recmo(out4)
        else:
            raise(RuntimeError("illegal model mode"))

        outup = self.f(self.fourup1(out))
        outup = self.f(self.fourup2(outup))
        out = self.outconv(outup)

        return out, prev, hi, acc#, test3



def FCNDUCLONG(num_labels=2, seqLength=1,mode='iid', **kwargs):
    """Constructs a dense kalman model. CONV-RNN on end with an initializer for the first frame in a sequence.

    Modifications by Will for Semantic Segmentation - added HDC in blocks 2 & 3 of values(block 0 & 1 left untouched):
    2: [1, 2, 3, 5, 9, 17],
    3: [5, 2, 1] #Based on experiments with TidalWave
    """
    #dil= {0: [1,1,1], 1: [1,1,1,1], 2:[1,2,3,5,9,17],3:[5,2,1]} #Based on experiments with TidalWave
    dil= {0: [(1,1),(1,1)], 1: [(1,1),(1,1),], 2:[(1,1),(1,1)],3:[(1,1),(2,2),(3,3),(1,1)], 4:[(1,1),(1,1)]} #Based on HDC
    model = FCNKalman(BasicBlockx, [2,2,2,4,2], dil, num_labels, mode)
    return model
