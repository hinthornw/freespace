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


__all__ = ['DenseShuffle23']




class DenseShuffle(nn.Module):

    def __init__(self, expansions=[3,3,3,1], dil={1:None, 2:None, 3:None, 4:None}, insize=486, num_labels=2):
        super(DenseShuffle, self).__init__()
        self.num_labels= num_labels
        self.outmultiple = expansions #[3, 3, 3, 1, 1#] 

        # Input Feature extractor
        self.convin = nn.Sequential( #Does adding nonlinearity help? -WILL
            nn.Conv2d(1, 16,7,2,3, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        layers = [4,6,10]
        theta = 2
        self.gr = 8
        self.inplanes = 16 
        self.planes = self.inplanes + layers[0]*self.gr
        self.block1 = DenseBlock(self.inplanes,layers[0],self.gr,[1,1,1,1])
        self.trans1 = TransitionLayer(self.planes, self.planes/theta, 2, True) 
        self.inplanes = self.planes/theta
        self.planes = self.inplanes + layers[1]*self.gr
        self.block2 = DenseBlock(self.inplanes,layers[1],self.gr,[1,1,2,3,1,1])
        self.trans2 = TransitionLayer(self.planes, self.planes/theta, 2, True) 
        self.inplanes = self.planes/theta
        self.planes = self.inplanes + layers[2]*self.gr
        self.block3 = DenseBlock(self.inplanes,layers[2],self.gr,[1,1,2,3,5,7,9,1,1,1])
        self.inplanes = 32 #self.planes / self.gr
        self.up = DUC(self.planes, self.inplanes, 16, 3,False, grouped=False)        
        #self.trans = nn.ConvTranspose2d(outwidth, 32, 3, 2, 1,output_padding=1, bias=False) 
      
        self.planes = self.num_labels
        self.smoothout = nn.Sequential(OrderedDict([
                  ( 'act1', nn.ELU(inplace=False)),
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
        x1 = self.trans1(x1)
        x2 = self.block2(x1)
        x2 = self.trans2(x2)
        x3 = self.block3(x2)
        xup = self.up(x3)
        xout = self.smoothout(xup)
        return xout






def printStats(out, name):
    print "{}:\t{}\t{}".format(name, out.data.max(), out.data.min())

def DenseShuffle23(num_labels=2, insize=448, **kwargs):
    """Constructs a leapnet model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseShuffle([3,3,3,1], insize=insize, num_labels=num_labels)
        
    return model
