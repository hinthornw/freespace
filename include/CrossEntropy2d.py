import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import time

import torch
from torch.autograd import Variable

CrossEntropyDebug_=False

class THNNLSM(nn.Module):
    def __init__(self):
        super (THNNLSM, self).__init__()
        #self.f = _functions.thnn.LogSoftmax()()
        self.f = F.log_softmax
    def forward(self, input):
        return self.f(input)


def CrossEntropy2d(prediction, target, weight=None, size_average=True, mask=None, deviceids=[0], name='', diff=None):
    global CrossEntropyDebug_
    torch.cuda.device(deviceids)
    n, c, h, w = prediction.size()
    freespace = 1
    if n > 1:
        raise(RuntimeError("Loss not defined for batch size > 1. Try accumulating gradient instead"))

    if mask is not None:
        prediction = prediction * mask.expand_as(prediction)
        target = target * mask.squeeze().long()

    resPrediction = None
    resTarget = None
    if CrossEntropyDebug_:
        values, indices = prediction.data.max(1)
        resPrediction = indices.cpu().numpy()
        resTarget = target.data.cpu().numpy()

    logsoftmax = THNNLSM()
    lossfn = nn.NLLLoss2d(size_average=False)

    prediction = logsoftmax(prediction)
    if CrossEntropyDebug_:
        print('MxMnMeSt:\t{}|{}|{}|{}|'.format(prediction.max().data[0],
                prediction.min().data[0], prediction.mean().data[0],prediction.std().data[0]))

    # BUGGY HERE: I convert ignore values to 0's so that they are thought to be OK
    ig_present = False
    jawn = prediction
    if target.min().data[0] < 0: #Are ignore values
        ig_present = True
        indices = ((target.view(h,w))< 0).view(1,1,h,w).data
        predictiona = prediction[0,0,:,:].clone().view(1,1,h,w)
        predictionb = prediction[0,1:,:,:].clone().view(1,c-1,h,w)
        predictiona[indices] = 0.0
#        predictionb[indices.repeat(1,c-1,1,1)] = -10.0 #irrelevant - doesnt look there
        prediction = torch.cat((predictiona, predictionb),1).contiguous()
        target[indices.view(h,w)] = 0
        target = target.view(n, h, w).contiguous()
    target = target.view(n, h, w).contiguous()
    loss = lossfn(prediction, target)
   # if ig_present:
   #     print "Loss with ig = {}".format(loss.data[0])
    #For now : Just set ignore to 0 and make sure you predict correctly
    # Note that this only works for batch size = 1
    
    if CrossEntropyDebug_:
        resPrediction = resPrediction.squeeze()
        I = np.float64(np.sum(np.logical_and(resPrediction== freespace, resTarget == freespace)))
        U = np.float64(np.sum(np.logical_or(np.logical_and(resPrediction == freespace, resTarget >=0), resTarget == freespace)))
        print 'Loss|IOU Road [ratio] =\t({}|{})\t[{}]'.format((I/U), loss.data[0], (I/U)/loss.data[0])
        #difficulty[name] = I/U


    return loss
