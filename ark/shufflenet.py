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


__all__ = ['ShuffleNet28']




class ShuffleNet(nn.Module):

    def __init__(self, expansions=[3,3,3,1], dil={1:None, 2:None, 3:None, 4:None}, insize=486, num_labels=2):
        super(ShuffleNet, self).__init__()
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
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1, bias=False))

        self.stemcomp = nn.Sequential(
                        nn.Conv2d(64, 32, 3,1,1,bias=False),
                        nn.ELU(inplace=True),
                        nn.Conv2d(32,32,3,1,1,bias=False),
                        nn.BatchNorm2d(32),
                        nn.Sigmoid())
        #Assuming we have an input sized at 486, we go down to 1x1 then densely combine
        #243->162
        self.planes = self.inplanes*3/2#*expansions[0]
        self.block1 = nn.Sequential(
                        #ShuffleBlock(self.inplanes,self.planes, 3),
                        GracefulDownsampling(self.inplanes,self.planes), 
                        SummaryBlock(self.planes, self.planes, dil[0]),
                        SummaryBlock(self.planes, self.planes, dil[0])) 
        #162->81
        self.inplanes = self.planes
        self.planes = self.inplanes*2 #expansions[1]
        self._2asize = self.planes
        self.block2a= nn.Sequential(
                        nn.ELU(inplace=True),
                        nn.Conv2d(self.inplanes, self.planes,3,2,1,bias=False),
                        nn.BatchNorm2d(self.planes),
                        nn.ELU(inplace=True),
                        SummaryBlock(self.planes, self.planes, dil[0]))
        #81->243
        self._2aup = nn.Sequential(
                        DUC(self._2asize, 32, 3, kernel_size=1),
                        nn.Conv2d(32, 32, 3,1,1,bias=False),
                        nn.ELU(inplace=True),
                        nn.Conv2d(32, 32, 3, 1, 1, bias=False))
        #81->54 
        self.inplanes = self.planes
        self.planes = self.inplanes*3/2 #expansions[1]
        self._2bsize = self.planes
        self.block2b= nn.Sequential(
                        GracefulDownsampling(self.inplanes,self.planes), 
                        SummaryBlock(self.planes, self.planes, dil[0]))
#                        SummaryBlock(self.planes, self.planes, dil[1])) 
        #54->81
        self._2bup = nn.Sequential(
#                            DDC(self._2bsize, 2),
                            nn.Conv2d(self._2bsize, self._2bsize*2,3,2,1,bias=False),
                            DUC(self._2bsize*2, self._2asize, 3, kernel_size=1),
                            nn.Conv2d(self._2asize,self._2asize,3,1,1,bias=False),
                            nn.ELU(inplace=True)
                        )
        #54->36
        self.inplanes = self.planes
        self.planes = self.inplanes*3/2#expansions[2]
        self._3size = self.planes
        self.block3 = nn.Sequential(
                        #ShuffleBlock(self.inplanes,self.planes, 3),
                        GracefulDownsampling(self.inplanes,self.planes), 
                        SummaryBlock(self.planes, self.planes, dil[2]))#,
#                        SummaryBlock(Self.planes, self.planes, dil[2]) 
        #36->81
        self._3up = nn.Sequential(
                        #DDC(self._3size, 4),
                        nn.Conv2d(self._3size, self._3size*2, 3,2,1,bias=False),
                        nn.ELU(inplace=True),
                        nn.Conv2d(self._3size*2,self._3size*4, 3,2,1,bias=False),
                        DUC(self._3size * 4, self._2asize,9),
                        nn.Conv2d(self._2asize, self._2asize,3,1,1,bias=False),
                        nn.ELU(inplace=True))
        #36->24
        self.inplanes = self.planes
        self.planes = self.inplanes*3/2 #*expansions[3]
        self._4size = self.planes
        self.block4 = nn.Sequential(
                        GracefulDownsampling(self.inplanes,self.planes), 
                        SummaryBlock(self.planes, self.planes))#,
             #           SummaryBlock(self.planes, self.planes))
        #24->22->20->18->54
        self._4up = nn.Sequential(
                        nn.ELU(inplace=True),
                        nn.Conv2d(self._4size, self._4size,3,1,0, bias=False),
                        nn.ELU(inplace=True),
                        nn.Conv2d(self._4size,self._4size,3,1,0,bias=False),
                        nn.ELU(inplace=True),
                        nn.Conv2d(self._4size,self._4size,3,1,0,bias=False),
                        nn.ELU(inplace=True),
                        DUC(self._4size,self._2bsize,3,kernel_size=1))


        #24->8->6
        self.inplanes = self.planes
        self.planes = self.inplanes*3/2 #*expansions[3]
        self._5size = self.planes
        self.block5 = nn.Sequential(
                        ShuffleBlock(self.inplanes,self.planes, 3, smooth=True),
                        nn.ELU(inplace=False),
                        nn.Conv2d(self.planes,self.planes,3,1,0,bias=False),
                        nn.BatchNorm2d(self.planes),
                        nn.ELU(inplace=True))
        #6->36
        self._5up = nn.Sequential(
                        DUC(self._5size, self._3size, 6, kernel_size=1),
                        nn.Conv2d(self._3size, self._3size,3,1,2,2,bias=False),
                        nn.ELU(inplace=True),
                        nn.Conv2d(self._3size, self._3size,3,1,1,1,bias=False))

       
        self.trans = nn.ConvTranspose2d(32, 32, 3, 2, 1,output_padding=1, bias=False) 
      
        self.inplanes = 32
        self.planes = self.num_labels
        self.smoothout = nn.Sequential(OrderedDict([
                  ( 'act1', nn.ELU(inplace=True)),
                  ( 'conv2', nn.Conv2d(self.inplanes, self.inplanes, 3, padding=1, bias=False)),
                  ( 'act' , nn.ELU(inplace=True)),
                  ( 'out'+str(self.num_labels),
                    nn.Conv2d(self.inplanes, self.planes, kernel_size=3, stride=1, padding=1))])) 
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x): 
        #Input = 486
        insize = x.size()
        xstem = self.convin(x) #64 x 243
        x1 = self.block1(xstem)#96 x 162
        x2a = self.block2a(x1) #192x81 
        x2b = self.block2b(x2a)#288 x 54
        x3 = self.block3(x2b) #432 x 36
        x4 = self.block4(x3) #648 x 24
        x5 = self.block5(x4) #972 x 6
        x5 = self._5up(x5) #432 x 36
        x3 = x3 + x5
        x3 = self._3up(x3) #192 x 81

        x4 = self._4up(x4) #288 x 54
        x2b = x2b + x4
        x2b = self._2bup(x2b) # 192 x 81

        x2a = x2a + x3 + x2b
        x2a = self._2aup(x2a) 
        stemout = self.stemcomp(xstem)
        stems = stemout * x2a
        xstem = self.trans(stems)
        xout = self.smoothout(xstem)
        return xout






def printStats(out, name):
    print "{}:\t{}\t{}".format(name, out.data.max(), out.data.min())

def ShuffleNet28(num_labels=2, insize=448, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    dil= {0: (1,1), 1: (1,2), 2:(1,1),3: (1,1)} #Based on HDC
    model = ShuffleNet([3,3,3,1], dil=dil, insize=insize, num_labels=num_labels)
        
    return model
