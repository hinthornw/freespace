#import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import time
#from collections import Queue
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



def distance_weights(label, scale = 10, dist_to=112): #dist_to=112 means weight maxes out ~112 pixels out or 1/4 of image size 
    '''Used to find height @ which the road stops (vertically seeking)'''
    dist_to = float(dist_to) /4.0
    h, w = label.size()
   # lb = np.zeros([1,w])
#    lb = np.zeros([2,w])
    label = label.transpose(0,1).contiguous()
    mask = np.zeros([w, h])
    def sig_array(x):
        return 1 / (1 + np.exp(-x))
    for i in range(w):
        edges = []
        lastedge= label[i,0]
        if lastedge == 0:
            edges.append(-1)
        else :
            edges.append(-500)
        for j in range(1,h): #Bottom of road
                if label[i,j] != lastedge:
                    edges.append(j-lastedge+1)
                    lastedge = label[i,j]
                    print "lastedge={}".format(lastedge),
        edges.append(447)
        print edges
        for j in range(h):
            k = 1
            if label[i,j] == 0:
                d = min(j-edges[k-1], edges[k]-j)
                print edges[k]
                print "min{}, {} = {}".format(j-edges[k-1], edges[k]-j, d)
                mask[i,j] == (float(d)+1.0)/dist_to 
            elif label[i,j] == 1:
                if j == edges[k]:
                    k+=1
                    mask[i,j]= 0.0 
                else:
                    d = min(j-edges[k-1], edges[k]-j)
                    mask[i,j] == (float(d)/dist_to)
    print mask
    mask = mask.transpose()      
    mask = (sig_array(mask) - 0.5) * scale 
    return torch.from_numpy(mask).float() 
def CrossEntropy2d(prediction, target, weight=None, size_average=True, mask=None, deviceids=[0], name='', diff=None):
    # prediction:(n, c, h, w) target:(n, h, w)
    global CrossEntropyDebug_
    torch.cuda.device(deviceids)
    n, c, h, w = prediction.size()
    #prediction = prediction.clone()
    #print "Target : {}".format(target)
    freespace = 1
    if n > 1:
        raise(RuntimeError("Loss not defined for batch size > 1. Try accumulating gradient instead"))
    if mask is not None:
        prediction = prediction * mask.expand_as(prediction)
        target = target * mask.squeeze().long()

    #prediction = prediction.permute(0,2,3,1).contiguous()
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
