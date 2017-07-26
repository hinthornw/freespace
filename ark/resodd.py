import sys
import torch.nn as nn
sys.path.append('./ark')
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import time
from layers import *



__all__ = ['ShuffleNet28']




class ShuffleNet(nn.Module):

    def __init__(self, expansions=[3,3,3,1], dilations={1:None, 2:None, 3:None, 4:None}, insize=486, num_labels=2):
        super(ResShuffle, self).__init__()
        self.inplanes = 64
        self.num_labels= num_labels
        self.outmultiple = expansions #[3, 3, 3, 1, 1] 
        self.f = nn.ELU(inplace=True) #nn.ReLU(inplace=True)

        # Input Feature extractor
        self.convin = nn.Sequential( #Does adding nonlinearity help? -WILL
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                                   bias=False),
            #nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1, bias=False))

        #Assuming we have an input sized at 486, we go down to 1x1 then densely combine
        #243->81
        self.planes = self.inplanes*expansions[0]
        self.block1 = nn.Sequential(
                        ShuffleBlock(self.inplanes,self.planes, 3),
                        SummaryBlock(self.planes, self.planes, dil[0])) 
        #81->27 
        self.inplanes = self.planes
        self.planes = self.inplanes*expansions[1]
        self.block2= nn.Sequential(
                        ShuffleBlock(self.inplanes,self.planes, 3)#,
#                        SummaryBlock(self.planes, self.planes, dil[1])) 
        #27->9
        self.inplanes = self.planes
       self.planes = self.inplanes*expansions[2]
        self.block3 = nn.Sequential(
                        ShuffleBlock(self.inplanes,self.planes, 3))#,
#                        SummaryBlock(self.planes, self.planes, dil[2])) 
        #9->3->1
        self.inplanes = self.planes
        self.planes = self.inplanes*expansions[3]
        self.block4 = nn.Sequential(
                        ShuffleBlock(self.inplanes,self.planes, 3, smooth=False),
                        nn.BatchNorm2d(self.planes))

#        self.inplanes = self.planes
#        self.planes - self.inplanes/expansion[3]
#        self.globalUp = DUC(self.inplanes, self.planes, 3, res=True)
        self.globalUp = nn.Upsample(scale_factor=9, mode='nearest') # merge straight w/ block2
        self.glob = nn.Sequential(
                            nn.Conv2d(self.planes, self.inplanes,3,1,0, bias=False),
                            nn.BatchNorm2d(self.inplanes),
                            nn.ELU()
                            self.globalUp)

        self.inplanes = self.planes
        self.planes = self.inplanes/expansions[3]
        self.DUC4_3 = DUC(self.inplanes, self.planes, 3, res=True) 
        self.smooth3_3 = nn.Sequential(
                            nn.Conv2d(self.planes, self.planes, 3,1,1, bias=False),
                            nn.BatchNorm2d(self.planes),
                            nn.ELU())

        
        self.inplanes = self.planes
        self.planes = self.inplanes/expansions[2]
        self.DUC3_2 = DUC(self.inplanes, self.planes, 3, res=True)
        self.smooth2_2 = nn.Sequential(
                            nn.Conv2d(self.planes, self.planes, 3,1,1, bias=False),
                            nn.BatchNorm2d(self.planes),
                            nn.ELU())


        self.inplanes = self.planes
        self.planes = self.inplanes/expansions[1]
        self.DUC2_1(self.inplanes, self.planes, 3, res=True)
        self.smooth1_1 = nn.Sequential(
                            nn.Conv2d(self.planes, self.planes, 3,1,1, bias=False),
                            nn.BatchNorm2d(self.planes),
                            nn.ELU())

        self.inplanes = self.planes
        self.planes = self.inplanes/expansions[0]
        self.DUC1_0(self.inplanes, self.planes, 3, res=True)
        self.smooth0_0 = nn.Sequential(
                            nn.Conv2d(self.planes, self.planes, 3,1,1, bias=False),
                            nn.BatchNorm2d(self.planes),
                            nn.ELU())
        
        self.trans = nn.ConvTranspose2d(self.inplanes, self.planes, kernel_size=3, stride=2, padding=1) 

        self.inplanes = self.planes
        self.planes = self.num_labels 
        self.smoothout = nn.Sequential(
                            nn.Conv2d(self.inplanes, self.inplanes, 3, padding=1),
                            nn.ELU(),
                            nn.Conv2d(self.inplanes, self.planes, kernel_size=3, stride=1, padding=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        #Input = 486
        xstem = self.convin(x) #64 x 243
        x1 = self.block1(xstem) #192 x 81
        x2 = self.block2(x1) #576 x 27
        x3 = self.block3(x2) #1728 x 9
        x4 = self.block4(x3) #1728 x3
        globalSum = self.glob(x4) #3->1->1728 x 9
        x4 = self.DUC4_3(x4) #1728 x 9
        residual = x4 + globalSum
        x4 = self.smooth3_3(residual + x3)
        x3 = self.DUC3_2(x4 + residual) # 576 x 27
        residual = x3 
        x3 = self.smooth2_2(x3 + x2)
        x2 = self.DUC2_1(residual + x3) # 192 x 81
        residual = x2
        x2 = self.smooth1_1(x2+x1)
        x1 = self.DUC1_0(residual+x2) # 64 x 243
        residual = x1
        x1 = self.smooth0_0(x1+xstem)
        x1 = self.trans(residual+x1) # 64 x 486
        x = self.smoothout(x1) # labels x 486 
        return x 

def ShuffleNet28(pretrained=False, stem='reg', insize=448, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    dil= {0: [(1,1), (1,1)], 1: [(1,1),(1,1)], 2:[(1,1),(1,1)],3:[(1,1), (1,1)]} #Based on HDC
    model = ShuffleNet([3,3,3,1], dilations=dil, insize=insize)
        
    return model
