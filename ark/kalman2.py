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

__all__ = ['FCNDUC']




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
        self.up2 = DUCSmooth(128, 2, 8,3,False,'ID')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dil=dilations[2])
        self.up3 = DUCSmooth(256, 2, 16,3,False,'ID')

        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dil=dilations[3])
        self.smooth = nn.Sequential(
                        nn.ELU(inplace=True),
                        nn.Conv2d(512,512,3,1,1,bias=False),
                        nn.ELU(inplace=True),
                        nn.Conv2d(512,512,3,1,1,bias=False))

        self.rec1 = nn.Conv2d(512, 512, 3,1,1,bias=False) 
        self.rec2 = nn.Conv2d(512, 512, 3,1,1,bias=False) 
        self.fourup1 = DUCSmooth(512, 2,16,3,False,'ID') 
        self.outconv = nn.Conv2d(2,2,3,1,1)
        self.fuse = nn.Conv2d(self.num_labels*2, self.num_labels,5,1,2)
        self.f = nn.ELU()
#        self.fourup1 = nn.Sequential(
#                        nn.BatchNorm2d(512),
#                        nn.ReLU(inplace=True),
#                        nn.ConvTranspose2d(512, 256, 3, 2, 1,output_padding=1, bias=False),
#                        nn.BatchNorm2d(256),
#                        nn.ReLU(inplace=True),
#                        nn.ConvTranspose2d(256, 128, 3, 2, 1,output_padding=1, bias=False),
#                        nn.BatchNorm2d(128),
#                        nn.ReLU(inplace=True),
#                        nn.ConvTranspose2d(128, 56, 3, 2, 1,output_padding=1, bias=False),
#                        nn.BatchNorm2d(56),
#                        nn.ReLU(inplace=True),
#                        nn.ConvTranspose2d(56, 2, 3, 2, 1,output_padding=1, bias=False))


        self.convoutControl = nn.Sequential(
                            nn.Conv2d(4,8,3,1,1,bias=False),
                            nn.BatchNorm2d(8),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(8,2,3,1,1))
##        self.convoutEntr = nn.Sequential(
#                            nn.Conv2d(4,8,3,1,1,bias=False),
#                            nn.BatchNorm2d(8),
#                            nn.ReLU(inplace=True),
#                            nn.Conv2d(8,2,3,1,1))


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
        #Out2 is size 56x56
        test2 = self.up2(out2)
        #entr2 = GetEntropy(test2)# B
        
        out3 = self.layer3(out2)
        #test3 = self.up3(out3)
#        entr3 = GetEntropy(test3) # G

        out4 = self.layer4(out3)
        out4 = self.smooth(out4)

        # Recurrent
        if 'End' in self.mode:
            out = self.fourup1(self.rec1(out))
            if self.mode == 'kalmanEnd':
                #Predictive Correlative Block
                b,c,h,w = out4.size()
                seq1 = []
                for i in range(b):
                    if i % self.seqLength == 0:
                        guess = cat([out[i].expand(1,c,h,w),out[i].expand(1,c,h,w)], 1)
                        seq1.append(self.fuse(guess))
                    else:
                        guess = cat([out[i].expand(1,c,h,w), seq1[i-1].expand(1,c,h,w)],1)
                        seq1.append(self.fuse(guess))
                out = torch.cat(seq1,0)
            elif self.mode == 'iidEnd':
                for i in range(b):
                    guess = cat([out[i].expand(1,c,h,w),out[i].expand(1,c,h,w)], 1)
                    seq1.append(self.fuse(guess))

            out = self.f(self.outconv(out))
        else:
            if self.mode == 'kalman':
                #Predictive Correlative Block
                b,c,h,w = out4.size()
                seq1 = []
                for i in range(b):
                    if i % self.seqLength == 0:
                        seq1.append(self.rec1(out4[i].expand(1,c,h,w)))
                    else:
                        seq1.append(seq1[i-1] + self.rec2(out4[i].expand(1,c,h,w)-out4[i-1].expand(1,c,h,w)))
                out = torch.cat(seq1,0)
            elif self.mode == 'iid':
                out = self.rec1(out4)
            outup = self.fourup1(out)
            out = self.outconv(outup)

        #outup = outup.detach()
        #test2 = test2.detach()
        #ch = outup.size(1)
        #entr = GetEntropy(out) # R
        #entr_ = entr
        ##for i in range(ch-1):
        #    entr = cat([entr, entr_], 1)
        #outupControl = cat([outup, test2],1).detach()
        #outControl = self.convoutControl(outupControl)

#       # outup = outup #* ()
        #test2 = test2 * (1.0-entr) # draw more info in difficult areas
        #outup = cat([outup, test2], 1)
        #outlast = self.convoutEntr(outup)

        #

        #entr2 = [None for i in range(b)] 
        #entr3 = [None for i in range(b)] 

        #print "{},".format(index),
        #for v in range(b):
        #    SaveFuzzyEntropy(x[v],entr_[v],'FCNDUC','layers_444_', index, entr2[v], entr3[v])
        #    index+=1


        return outlast, outControl, out #, test3



def FCNDUC(num_labels=2, seqLength=1,mode='iid', **kwargs):
    """Constructs a dense kalman model. CONV-RNN on end with an initializer for the first frame in a sequence.

    Modifications by Will for Semantic Segmentation - added HDC in blocks 2 & 3 of values(block 0 & 1 left untouched):
    2: [1, 2, 3, 5, 9, 17],
    3: [5, 2, 1] #Based on experiments with TidalWave
    """
    #dil= {0: [1,1,1], 1: [1,1,1,1], 2:[1,2,3,5,9,17],3:[5,2,1]} #Based on experiments with TidalWave
    dil= {0: [(1,1),(1,1)], 1: [(1,1),(1,1),], 2:[(1,1),(1,1)],3:[(1,1),(2,2),(3,3),(1,1)], 4:[(1,1),(1,1)]} #Based on HDC
    model = FCNKalman(BasicBlockx, [2,2,2,4,2], dil, num_labels, mode)
    return model
