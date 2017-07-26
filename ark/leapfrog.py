import sys
import torch.nn as nn
sys.path.append('./ark')
import math
from torch import cat
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import time
from layers import *
from collections import OrderedDict


__all__ = ['LeapNet7']




class LeapNet(nn.Module):

    def __init__(self, expansions=[3,3,3,1], dil={1:None, 2:None, 3:None, 4:None}, insize=486, num_labels=2):
        super(LeapNet, self).__init__()
        self.inplanes = 64
        self.num_labels= num_labels
        self.outmultiple = expansions #[3, 3, 3, 1, 1#] 
#        self.f = nn.ELU(inplace=True) #nn.ReLU(inplace=True)

        # Input Feature extractor
        self.convin = nn.Sequential( #Does adding nonlinearity help? -WILL
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                                   bias=False),
            #nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
#            nn.ELU(inplace=True),
#            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64, kernel_size=3, stride=2, padding=1, bias=False))
        outwidth = 64

        down = [2,3,4,6,9,12,18,27]
        exp =  [2,3,4,6,9,9,9,6]
        self.block1 = LeapBlock(outwidth,exp[0],down[0],True,1,1,(False,False))
        self.block2 = LeapBlock(outwidth,exp[1],down[1],True,1,1,(False,False))
        self.block3 = LeapBlock(outwidth,exp[2],down[2],True,1,1,(False,False))
        self.block4 = LeapBlock(outwidth,exp[3],down[3],True,1,1,(False,False))
        self.block5 = LeapBlock(outwidth,exp[4],down[4],True,1,1,(False,False))
#        self.block6 = LeapBlock(outwidth,exp[5],down[5],True,1,1,(False,False))
        #self.block7 = LeapBlock(outwidth,exp[6],down[6],True,1,1,(False,False))
#        self.block8 = LeapBlock(outwidth,exp[7],down[7],0,1,(False,False))
        self.up = DUC(outwidth, 32, 4, 3,False, grouped=True)        
        #self.trans = nn.ConvTranspose2d(outwidth, 32, 3, 2, 1,output_padding=1, bias=False) 
      
        self.inplanes = 32
        self.planes = self.num_labels
        self.smoothout = nn.Sequential(OrderedDict([
                  ( 'act1', nn.ELU(inplace=True)),
                  ( 'conv2', nn.Conv2d(self.inplanes, self.planes, 3, padding=1, bias=False))]))#,
       #           ( 'act' , nn.ELU(inplace=True)),
       #           ( 'out'+str(self.num_labels),
       #             nn.Conv2d(self.inplanes, self.planes, kernel_size=3, stride=1, padding=1))])) 
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x): 
        #Input = 432
        xstem = self.convin(x)
        x1 = self.block1(xstem)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
#        x6 = self.block6(x5)
        #x7 = self.block7(x6)
    #    x8 = self.block8(x7)
        xup = self.up(x5)
        xout = self.smoothout(xup)
        return xout






def printStats(out, name):
    print "{}:\t{}\t{}".format(name, out.data.max(), out.data.min())

def LeapNet7(num_labels=2, insize=448, **kwargs):
    """Constructs a leapnet model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LeapNet([3,3,3,1], insize=insize, num_labels=num_labels)
        
    return model
