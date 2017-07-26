import sys
import torch.nn as nn
sys.path.append('./ark')
import math
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
        self.outmultiple = expansions #[3, 3, 3, 1, 1] 
        self.f = nn.ELU(inplace=True) #nn.ReLU(inplace=True)

        # Input Feature extractor
        self.convin = nn.Sequential( #Does adding nonlinearity help? -WILL
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                                   bias=False),
            #nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
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
                        ShuffleBlock(self.inplanes,self.planes, 3),
                        SummaryBlock(self.planes, self.planes, dil[1])) 
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
#        self.globalUp = DUC(self.planes, self.planes/expansions[3], 9, res=False, kernel_size=1)
#        self.globalUp = nn.Upsample(scale_factor=9, mode='nearest') # merge straight w/ block2
#        self.globalUp = UpsamplingNearest2d(size=None, scale_factor=9)
        self.globalUp = nn.ReplicationPad2d(4) # copy it all :P
        self.glob = nn.Sequential(
                            nn.Conv2d(self.planes, self.planes,3,1,0, bias=False),
                            nn.BatchNorm2d(self.planes),
                            nn.ELU(inplace=True),
                            self.globalUp,
                            nn.BatchNorm2d(self.planes))


        self.inplanes = self.planes
        self.planes = self.inplanes/expansions[3]
        self.DUC4_3 = DUC(self.inplanes, self.planes, 3, res=True) 
##        self.smooth3_3 = nn.Sequential(
#                            nn.Conv2d(self.planes, self.planes, 3,1,1, bias=False),
#                            nn.BatchNorm2d(self.planes),
#                            nn.ELU(inplace=True))

        smooth_dil = (1,1)
        self.smooth3_3 = SummaryBlock(self.planes, self.planes,smooth_dil )        
        self.inplanes = self.planes
        self.planes = self.inplanes/expansions[2]
        self.DUC3_2 = DUC(self.inplanes, self.planes, 3, res=True)
#        self.smooth2_2 = nn.Sequential(
#                            nn.Conv2d(self.planes, self.planes, 3,1,1, bias=False),
#                            nn.BatchNorm2d(self.planes),
#                            nn.ELU(inplace=True))
        self.smooth2_2 = SummaryBlock(self.planes, self.planes,smooth_dil)

        self.inplanes = self.planes
        self.planes = self.inplanes/expansions[1]
        self.DUC2_1 = DUC(self.inplanes, self.planes, 3, res=True)
#        self.smooth1_1 = nn.Sequential(
#                            nn.Conv2d(self.planes, self.planes, 3,1,1, bias=False),
#                            nn.BatchNorm2d(self.planes),
#                            nn.ELU(inplace=True))
        self.smooth1_1 = SummaryBlock(self.planes, self.planes, smooth_dil)

        self.inplanes = self.planes
        self.planes = self.inplanes/expansions[0]
        self.DUC1_0 = DUC(self.inplanes, self.planes, 3, res=True)
#        self.smooth0_0 = nn.Sequential(
#                            nn.Conv2d(self.planes, self.planes, 3,1,1, bias=False),
#                            nn.BatchNorm2d(self.planes),
#                            nn.ELU(inplace=True))
        self.smooth0_0 = SummaryBlock(self.planes, self.planes, smooth_dil)
        
        self.trans = nn.ConvTranspose2d(self.planes, self.planes, kernel_size=3, stride=2, padding=1, bias=False) 

        self.inplanes = self.planes
        self.planes = self.num_labels 
        #Make s](moothout a dictionary object to trick the weight transfers
        self.smoothout = nn.Sequential(OrderedDict([
                  ( 'conv2', nn.Conv2d(self.inplanes, self.inplanes, 3, padding=1, bias=False)),
                  ( 'act' , nn.ELU(inplace=True)),
                  ( 'out'+str(self.num_labels),
                    nn.Conv2d(self.inplanes, self.planes, kernel_size=3, stride=1, padding=1))])) 
        #self.smoothout = {self.num_labels: nn.Sequential(
        #                    nn.Conv2d(self.inplanes, self.inplanes, 3, padding=1, bias=False),
        #                    nn.ELU(inplace=True),
        #                    nn.Conv2d(self.inplanes, self.planes, kernel_size=3, stride=1, padding=1))}
        print "Initializing Parameters"
#        i = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #    print type(m)
        #    i = i + 1
#            if i % 5 == 0:
#                print "{} modules initialized".format(i)


    def forward(self, x): 
        #Input = 486
        insize = x.size()
        xstem = self.convin(x) #64 x 243
        istraining = False #stem.training 
        if not istraining:
            printStats(xstem, 'xstem layer')
        x1 = self.block1(xstem) #192 x 81

        if not istraining:
            printStats(x1, 'x1 layer')
        x2 = self.block2(x1) #576 x 27
        if not istraining:
            printStats(x2, 'x2 layer')
        x3 = self.block3(x2) #576 x 9
        if not istraining:
            printStats(x3, 'x3 layer')
        x4 = self.block4(x3) #576 x 3
        if not istraining:
            printStats(x4, 'x4 layer')
        print x4
    
        globalSum = self.glob(x4) #3->1->576 x 9
        print "PRINTING WEIGHTS FOR GLOB"
        for m in self.glob.modules():
            if isinstance(m, nn.Conv2d):
                print "{}".format(m.weight.data)
        if not istraining:
            printStats(globalSum, 'globalSum layer')
        print globalSum
        print "TOTAL ZEROS"
        print "{}".format((globalSum == 0).data.sum())
#        printStats(globalSum, "Global Sum")
#        print "First few globalsum:\n{}".format(globalSum[0,0:5,:,:])
        x4 = self.DUC4_3(x4) #576 x 9
        if not istraining:
            printStats(x4, 'DUCx4 layer')
#        print "GS|x4: {}, {}".format(globalSum.size(), x4.size())
#        residual = x4 + globalSum
        x4 = self.smooth3_3(x4+globalSum) # + x3)
        if not istraining:
            printStats(x4, 'smooth x4 layer')
        x3 = self.DUC3_2(x4) # + residual) # 576 x 27
        #residual = x3 
        if not istraining:
            printStats(x3, 'DUCx3 layer')
        x3 = self.smooth2_2(x3 + x2)
        if not istraining:
            printStats(x3, 'Smooth x3 layer')
        x2 = self.DUC2_1(x3)  # + residual) # 192 x 81
        if not istraining:
            printStats(x2, 'DUCx2 layer')
#        residual = x2
        x2 = self.smooth1_1(x2+x1)
        if not istraining:
            printStats(x2, 'Smoothx1 layer')
        x1 = self.DUC1_0(x2)# + residual) # 64 x 243
        if not istraining:
            printStats(x1, 'DUCx1 layer')
#        residual = x1
        x1 = self.smooth0_0(x1+xstem)
        if not istraining:
            printStats(x1, 'Smoothx0 layer')
        x1 = self.trans(x1, output_size=insize) # 64 x 486
        if not istraining:
            printStats(x1, 'Transposed layer')
        #smoothout = self.smoothout[self.num_labels].cuda()
        x = self.smoothout(x1) # labels x 486 
        if not istraining:
            printStats(x, 'output layer')
            print 'Prediction:\n{}'.format(x.data.max(1)[1])
        print x
        exit()
        return x 

def printStats(out, name):
    print "{}:\t{}\t{}".format(name, out.data.max(), out.data.min())

def ShuffleNet28(num_labels=2, insize=448, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    dil= {0: (1,1), 1: (1,2), 2:(1,1),3: (1,1)} #Based on HDC
    model = ShuffleNet([3,3,3/2,1], dil=dil, insize=insize, num_labels=num_labels)
        
    return model
