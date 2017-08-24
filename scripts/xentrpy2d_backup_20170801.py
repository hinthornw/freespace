#import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import time

import torch

CrossEntropyDebug_= False 

class THNNLSM(nn.Module):
    def __init__(self, temp=1):
        super (THNNLSM, self).__init__()
        #self.f = _functions.thnn.LogSoftmax()()
        self.f = F.log_softmax
        self.temp = float(temp)
    def forward(self, input):
        return self.f(input/self.temp)
class nllloss(nn.Module):
    def __init__(self, ig=-1):
        super(nllloss, self).__init__()
        self.ig = ig
    def forward(self, prediction, target):
        loss = F.nll_loss(prediction, target, size_average=False, ignore_index=self.ig)
#        print "Loss: {}".format(loss.data)
        return loss

def CrossEntropy2d(prediction, target, weight=None, size_average=True, mask=None, deviceids=[0], name='', diff=None):
    global CrossEntropyDebug_
    torch.cuda.device(deviceids)
    n, c, h, w = prediction.size()

    freespace = 1
#    print '{}\t{}'.format(prediction.data.max(),prediction.data.min())

    if mask is not None:
        print "Applying mask"
        prediction = prediction * mask.expand_as(prediction)
        target = target * mask.squeeze().long()

    resPrediction = None
    resTarget = None
    if CrossEntropyDebug_:
        values, indices = prediction.data.max(1)
        resPrediction = indices.cpu().numpy()
        resTarget = target.data.cpu().numpy()

    logsoftmax = THNNLSM(temp=1)
#    weights = torch.ones(c+1)

#    weights[c] = 0.0 #Ignore Label
    
    #lossfn = nn.NLLLoss2d(weights, size_average=False)
#    lossfn = nn.NLLLoss(size_average=False, ignore_index=-1)

    ind = (target< 0)
    tot = ind.sum()
    #if ind.sum() > 0:
#    weights = torch.ones(c).cuda(deviceids[0])
    ig_ = c
    if tot.data[0] > 0:
#        print "Ignore labels found"
#        weights = torch.ones(c+1).cuda(deviceids[0])
#        print "MINMAX: ({}, {})".format(target.data.min(), target.data.max())
        extra = Variable((torch.ones(n,1,h,w)*-100000.0).cuda(deviceids[0]))
        c += 1
        ls = [prediction, extra]
        prediction = torch.cat(ls, 1)
#        print prediction
    prediction = logsoftmax(prediction)
#    if CrossEntropyDebug_:
#        print('MxMnMeSt:\t{}|{}|{}|{}|'.format(prediction.max().data[0],
#                prediction.min().data[0], prediction.mean().data[0],prediction.std().data[0]))

#    # BUGGY HERE: I convert ignore values to 0's so that they are thought to be OK
#    ig_present = False
#    jawn = prediction

#    if target.min().data[0] < 0: #Are ignore values
#    #    print "Ignoring values"
#        ig_present = True
#
#        indices = (target >= 0).view(n,1,h,w).data.contiguous()
#        mask = (prediction * 0.0) + 1.0
#        print "INDICIES: {}".format(200704-np.sum(indices.cpu().numpy()))
#        print "MASK: {}".format(mask)
#        print type(indices)
#        print indices.long()
#        print mask[:,0,:,:]()
#        mask[:,0, :, :] = mask[:, 0,:, :]* indices.long()
#        prediction *= mask.long() 
#        #predictiona = prediction[:,0,:,:].clone().view(n,1,h,w)
#        #predictionb = prediction[:,1:,:,:].clone().view(n,c-1,h,w)
#        #predictiona[indices] = 0.0
#        #prediction = torch.cat((predictiona, predictionb),1).contiguous()
#        target[indices.view(n,h,w)] = 0
#        #target = target.view(n, h, w).contiguous()

    target = target.view(n, h, w).contiguous().detach()
#    print "Sizes for lossfn"
#    print "{}\t{}".format(prediction.size(), target.size())
#    print "devices:\t{}".format(torch.cuda.current_device())
#    print "Stats:\n{}, {}\n{}, {}".format(prediction.max(), prediction.min(),target.max(), target.min()) 
#    loss = lossfn(prediction, target)
#    print F.nll_loss

#    ind = ind.detach()
    targ = target.data
    targ[ind.data] = ig_
    target = Variable(targ)

#    target[ind] = ig_
#    ind = ind.detach()
#    target = target.detach()
#    weights[freespace]*=2 #Double importance of recall
#    weights = weights.cuda(deviceids[0])
    #loss = F.nll_loss(prediction, target, weights, False, ig_)
    lossf = nllloss(ig_)
    mini_, maxi_ = target.data.min(), target.data.max() 
    classes = prediction.size(1)
#    print "({},{})".format(classes, target.size()),
    if mini_< 0 or maxi_ > c:
        print "({}, {})".format(mini_, maxi_)
    loss = lossf(prediction, target)
#    print "\t({})".format(loss.data[0])
    #For now : Just set ignore to 0 and make sure you predict correctly
    # Note that this only works for batch size = 1
    
    if CrossEntropyDebug_:
        resPrediction = resPrediction.squeeze()
        I = np.float64(np.sum(np.logical_and(resPrediction== freespace, resTarget == freespace)))
        U = np.float64(np.sum(np.logical_or(np.logical_and(resPrediction == freespace, resTarget >=0), resTarget == freespace)))
#        print "%Freespace:\t{}\t{}\t{}".format(float(np.sum(resTarget))/200704.0, I, U)
        if U <= 0:
            U = 1.0
        print 'IOU/Loss Road [ratio] =({}/{})\t({}|{})\t[{}]'.format(I, U, (I/U), loss.data[0], ((I/U)*10000.0)/loss.data[0])
        #difficulty[name] = I/U

    return loss





