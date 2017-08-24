import sys
import torch.nn as nn
import math
sys.path.append('./ark')
import math
from torch import cat, zeros
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

import time
from layers import *
from collections import OrderedDict


__all__ = ['Tiramisu40']




class Tiramisu(nn.Module):

    def __init__(self,dil={1:None, 2:None, 3:None, 4:None}, insize=486, num_labels=2, recmode=False, deviceids=0):
        super(Tiramisu, self).__init__()
        self.recmode = recmode
        self.num_labels= num_labels

        self.inplanes = 48 
        # Input Feature extractor
        self.convin = nn.Sequential( #Does adding nonlinearity help? -WILL
            nn.Conv2d(1, self.inplanes,7,1,3, bias=False),
#            nn.ELU(inplace=True),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True))#,
#            nn.Conv2d(16, 16, 3, 2, 1, bias=False),
#            nn.BatchNorm2d(16),
#            nn.ReLU(inplace=True))
                           
        layers = [2,2,2,5,12]
        theta = 0.6#0.75#3/4 
        self.gr = 12


        self.planes = self.inplanes + layers[0]*self.gr
        self.out1 = self.planes
        self.block1 = DenseBlock(self.inplanes,layers[0],self.gr,[1,1])
        self.trans1 = TransitionLayer(self.planes, int(math.ceil(self.planes*theta)), 2, True) 

        self.inplanes = int(math.ceil(self.planes*theta))
        self.planes = self.inplanes + layers[1]*self.gr
        self.out2 = self.planes
        self.block2 = DenseBlock(self.inplanes,layers[1],self.gr,[1,1])
        self.trans2 = TransitionLayer(self.planes, int(math.ceil(self.planes*theta)), 2, True) 

        self.inplanes = int(math.ceil(self.planes*theta))
        self.planes = self.inplanes + layers[2]*self.gr
        self.out3 = self.planes
        self.block3 = DenseBlock(self.inplanes,layers[2],self.gr,[1,1])
        self.trans3 = TransitionLayer(self.planes, int(math.ceil(self.planes*theta)), 2, True) 

        self.inplanes = int(math.ceil(self.planes*theta))
        self.planes = self.inplanes + layers[3]*self.gr
        self.out4 = self.planes
        self.block4 = DenseBlock(self.inplanes,layers[3],self.gr,[1,1,2,3,1])
        self.trans4 = TransitionLayer(self.planes, int(math.ceil(self.planes*theta)), 2, True) 

        self.inplanes = int(math.ceil(self.planes*theta))
        self.planes = self.inplanes + layers[4]*self.gr
        self.block5 = DenseBlock(self.inplanes, layers[4], self.gr, [1,1,2,2,3,3,3,1,1,1,1,1])
        self.rnn_hidden_size = self.inplanes *2
        self.rnn_h, self.rnn_w = 28,28
        self.b = 1
        if self.recmode in [1,2,5,6]:
            self.rec = ConvElman(self.inplanes, self.rnn_hidden_size,3)
        elif self.recmode in [3,4,7]:
            self.rec = ConvElman(self.inplanes, self.rnn_hidden_size,3,1,1,'mult')
        elif self.recmode in [8]:
            self.rec = ConvElman(self.inplanes, self.rnn_hidden_size,3,1,1,'mult', 'gatedID')
        elif self.recmode in [9]:
            self.rec = ConvElman(self.inplanes, self.rnn_hidden_size,3,1,1,'mult', 'gatedELU')


        self.inplanes = self.planes + self.out4 
        self.up5 = TransitionUp(self.planes, self.planes)  
        self.block6 = DenseBlock(self.inplanes, layers[3], self.gr,[1,1,2,3,1], False)


        self.planes = self.gr * layers[3]
        self.inplanes = self.planes + self.out3
        self.up4 = TransitionUp(self.planes, self.planes)
        self.block7 = DenseBlock(self.inplanes, layers[2], self.gr, [1,1], False)

        self.planes = self.gr * layers[2]
        self.inplanes = self.planes + self.out2
        self.up3 = TransitionUp(self.planes, self.planes)
        self.block8 = DenseBlock(self.inplanes, layers[1], self.gr, [1,1], False)


        self.planes =  self.gr * layers[1] #self.inplanes
        self.inplanes = self.planes #+ self.out1
        self.up2 = TransitionUp(self.planes, self.planes)
        self.block9 = DenseBlock(self.inplanes, layers[0], self.gr, [1,1], False)
        self.planes =  self.gr * layers[0]# self.planes 
        self.out = nn.Conv2d(self.planes, self.num_labels, 3,1,1)
#        self.outparam = nn.Sequential(
#                        nn.BatchNorm2d(self.num_labels),
#                        nn.ReLU(inplace=True),
#                        nn.Conv2d(self.num_labels, self.num_labels, (10,1),(8,1),(1,0), bias=True),
#                        nn.BatchNorm2d(self.num_labels),
#                        nn.ReLU(inplace=True),
#                        nn.Conv2d(self.num_labels, 1, (10,1),(8,1),(1,0),bias=False),
#                        nn.Conv2d(1,1,(7,1),(1,1),(0,0),bias=True))

#        self.vertMask = L1VertMask(1, 448,448, deviceids=deviceids)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def init_hidden(self):
        b, c, h, w = self.b, self.rnn_hidden_size, self.rnn_h, self.rnn_w
        if self.recmode == 0 or self.recmode == 1 or self.recmode == 2:
            return Variable(zeros(b, c, h, w))
        else:
            return Variable(ones(b,c,h,w))

    def recurse(self, xr, h, i):
        if self.recmode > 0:
            y = None
#            if i == 0:
#                y, h = self.rec(xr, xr)
#            else:
            y, h = self.rec(xr, h)
            if i > 0:
                if self.recmode in [1,3,5]: #xt = xt' + f(g(xt') + h(x[t-1]))
                    return xr + y
                elif self.recmode in [2,4]: #xt = xt'*f(g(xt') + h(x[t-1]))
                    return xr * y
                elif self.recmode in [6,7,8,9,10]:
                    return xr - y
        return xr

    def forward(self, x, h, i): 
        #Input = 432
#        print "{}: {}".format(i, x.size())
        xstem = self.convin(x)
        x1 = self.block1(xstem)
        x2 = self.block2(self.trans1(x1))
        x3 = self.block3(self.trans2(x2))
        x4 = self.block4(self.trans3(x3))
        xr = self.trans4(x4)
        if self.recmode < 10 and self.recmode > 0:
            xr = self.recurse(xr,h,i)
        if self.recmode > 0:
            y = None
#            if i == 0:
#                y, h = self.rec(xr, xr)
#            else:
            y, h = self.rec(xr, h)
            if i > 0:
                if self.recmode in [1,3,5]: #xt = xt' + f(g(xt') + h(x[t-1]))
                    xr = xr + y
                elif self.recmode in [2,4]: #xt = xt'*f(g(xt') + h(x[t-1]))
                    xr = xr * y
                elif self.recmode in [6,7,8,9]:
                    xr = xr - y
        x5 = self.block5(xr)
#        print "x5: {}".format(x5.size())
        x5up = cat([self.up5(x5),x4],1)
        x5up = self.block6(x5up)
#        print "block6: {}".format(x5up.size())
        x4up = cat([self.up4(x5up),x3],1)
        x4up = self.block7(x4up)
#        print "block7: {}".format(x4up.size())
        x3up = cat([self.up3(x4up),x2],1)
        x3up = self.block8(x3up)
#        print "block8: {}".format(x3up.size())
        x2up = self.up2(x3up)
#        x2up = cat([self.up2(x3up),x1],1)
        x2up = self.block9(x2up)
#        print "block 9 {}".format(x2up.size())
        xout = self.out(x2up)
#        xparam = self.outparam(xout.detach())

#        print "xParam: ({}), shp: {}, max: {}".format(xparam.mean().data[0], xparam.size(), xparam.max().data[0])
    #    print "Params: {}".format(xparam)
#        print "1:({})".format(xout[0,1,10,10]),
#        if self.training:
#            xout = self.vertMask(xout, xparam)
#        print "\t2:({})".format(xout[0,1,10,10])
        return xout, h #, xparam






def printStats(out, name):
    print "{}:\t{}\t{}".format(name, out.data.max(), out.data.min())

def Tiramisu40(num_labels=2, insize=448, recmode=False, deviceids=0, **kwargs):
    """Constructs a leapnet model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Tiramisu(insize=insize, num_labels=num_labels, recmode=recmode, deviceids=deviceids)
        
    return model
