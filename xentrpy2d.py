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
        return loss

def CrossEntropy2d(prediction, target, weight=None, size_average=True, mask=None, deviceids=[0], name='', diff=None):
    global CrossEntropyDebug_
    torch.cuda.device(deviceids)
    n, c, h, w = prediction.size()
    target = Variable(target.data.clone())

    #print "Max: ({}, {})".format(target.data.min(),target.data.max()),


    freespace = 1

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




    ind = (target< 0).data
    tot = ind.sum()

    ig_ = c
#    print "tot: {}".format(tot)
    lossf = nllloss(c)
    if tot > 0:
    #    print "Ig found"
        extra = Variable((torch.ones(n,1,h,w)*-10000000.0).cuda(deviceids[0]))
        ls = [prediction, extra]
        prediction = torch.cat(ls, 1)
        c+=1
        lossf = nllloss(ig_)

    logsoftmax = THNNLSM(temp=1)
    prediction = logsoftmax(prediction)

    #Create a legal predication
    target = target.view(n, h, w).contiguous().detach()
    targ = target.data
    targ[ind] = ig_
    target = Variable(targ)



   # ind = (target.data == freespace).cpu().long()
   # mask = torch.ones(n,c,h,w).cuda(deviceids[0])

   # for i in range(n):
   #     ind = (target[i].data == freespace)
   #     mask[i,:,ind] = 2.0
   # 

   # mask[,ind] = 2
   # mask = Variable(mask)
   # prediction = prediction * mask

    loss = lossf(prediction, target)
    
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





