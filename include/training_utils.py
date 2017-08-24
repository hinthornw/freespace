import os
import torch.nn as nn
import torch
import numpy as np
import cv2
from collections import deque









def savefuzzyoutput(im, pred, targ, index, dims=448, savename = '', q='AWFUL', modelname=''):
    root = '/data/hinthorn/workspace_hinthorn/exp/pytorch'
    savename = os.path.join(root, 'imgs','fuzzy_'+modelname +'_'+ savename+'_epoch_'+ str(index)+'_'+q + '.png')
    print "Saving: {}".format(savename)
    pred = torch.nn.functional.softmax(pred)*200
    pred = pred.data.cpu().squeeze().numpy()
    target = targ.data.cpu().squeeze().numpy()
    im = im.squeeze().numpy()
    im = (im - np.min(im))
    im = (im / np.max(im)) * 150
#    pred = pred.argmax(0)#[1]
    #pred = (pred - np.min(pred))
    #pred = (pred / np.max(pred))*230

        
    temp = np.zeros([dims, dims, 3], dtype=np.float32)
    for i in range(0,3):
        temp[:,:, i] = im
    im = temp
    alpha = 0.3
    im[:,:, 1] = im[:,:,1] * alpha + (1-alpha) * pred[0,:,:] #Blue
    im[:,:, 0] = im[:,:,0] *alpha + (1-alpha) * pred[1,:,:] #Greed
    ind = target == 1
    im[ind, 2] = im[ind,2] *alpha + (1-alpha) * 150 
    im = np.array(im, dtype=np.uint8) 
    cv2.imwrite(savename, im)



def jaccard(output, label,num_labels=2):
    output = output.data
    label = label.data
    values, predictmap = output.max(1)
    predictmap = predictmap.cpu().byte().squeeze().numpy()
    lab = label.cpu().byte().squeeze().numpy()
    IOU = np.zeros(num_labels, dtype=np.float32)
    I = np.zeros(num_labels, dtype=np.float32)
    U = np.zeros(num_labels, dtype=np.float32)
    REC = np.zeros(num_labels, dtype=np.float32)
    PREC = np.zeros(num_labels, dtype=np.float32)
    smoothing_const = 0.000001
    for j in range(num_labels):
        I[j] = np.float64(np.sum(np.logical_and((predictmap == j),(lab == j))))
        U[j] = np.float64(np.sum(np.logical_or(np.logical_and(predictmap == j, lab != 255), lab == j)))
        PREC[j] = (I[j]+smoothing_const)/(np.sum(predictmap == j)+smoothing_const)
        REC[j] = (I[j]+smoothing_const)/(np.sum(lab == j)+smoothing_const) 

        #ignore appropriate pixels
        if U[j] == 0: 
#                I[j] = 1
#                U[j] = 1 
            IOU[j] = -1
        else:
            IOU[j] =I[j]/U[j] 
    return IOU, PREC, REC, I, U    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageWindow(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=10, init=0):
        self.maxlength = length
        self.avg_ = init
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = self.avg_
        self.items = deque(maxlen=self.maxlength)
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        if len(self.items) == self.maxlength:
            self.sum -= self.items.popleft()
        else:
            self.count += 1
        self.items.append(val)
        self.avg = self.sum / self.count

def saveoutput(im, pred, targ, index,dims=448, savename = '', q='AWFUL', modelname=''):
    root = '/data/hinthorn/workspace_hinthorn/exp/pytorch'
    savename = os.path.join(root, 'imgs',modelname +'_'+ savename+'_epoch_'+ str(index)+'_'+q + '.png')
    pred = pred.data.cpu().squeeze().numpy()
    target = targ.data.cpu().squeeze().numpy()
    im = im.squeeze().numpy()
    im = (im - np.min(im))
    im = (im / np.max(im)) * 150
    pred = pred.argmax(0)#[1]
        
    temp = np.zeros([dims, dims, 3], dtype=np.float32)
    for i in range(0,3):
        temp[:,:, i] = im
    im = temp
    alpha = 0.3
    ind = pred == 1
    im[ind, 1] = im[ind,1] * alpha + (1-alpha) * 230
    ind = target == 1
    im[ind, 2] = im[ind,2] *alpha + (1-alpha) * 200
    im = np.array(im, dtype=np.uint8) 
    #print "Max image after: {}".format(np.max(im))
    cv2.imwrite(savename, im)
