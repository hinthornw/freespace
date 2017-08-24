import sys
sys.path.append('./include')
sys.path.append('./ark')
sys.path.append('./scripts')
from rnn_loader import RNNLoader 
# from include/CrossEntropy2d import CrossEntropy2d
import argparse
import shutil
import time
import os
import torch
from collections import deque
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import cv2
from losses import *
#from CrossEntropy2d import CrossEntropy2d
from xentrpy2d import CrossEntropy2d 
from res18 import *
from res50_dilated import *
from res50_recurrent import *
from fetch_model import getModelParams
print_freq = 200
calc_freq = 40 
batch_size = 1#15
workers = 8
epochs = 50 
deviceids = [4]
base_lr = 1e-2 
gradient_clip = 1
start_epoch = 0
sel_type = ("set", 10)
sel_type_val = ("set", 5)
#hard_boost = True
modelname = ''
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--resume', '-r', action="store", default=0, type=int)
parser.add_argument('--basemodel', '-b', action="store", default='resrecElm50', type=str)
parser.add_argument('--gpu', '-g', action="store", default="0", type=str)
parser.add_argument('--sequence', '-s', action="store", default=5, type=int)
parser.add_argument('--dims', '-d', action="store", default=448, type=int)
parser.add_argument('--labels', '-l', action='store', default=2, type=int)
parser.add_argument('--which', '-w', action='store', default='kitti', type=str)
parser.add_argument('--valfirst', '-v', action='store', default=0, type=int)
parser.add_argument('--From', '-f', action='store', default=0, type=int)
num_labels = 2
accum_batch = 30 #15 #10
save_validation = True 
min_save = -2 
save_training = True 
val_first = False 
#val_first = True 

dims = 448
dataset = 'mapcar'
def main():
    global start_epoch, args, save_name, best_name, num_labels, modelname, deviceids, val_first, dims, dataset

    args = parser.parse_args()
    bestIOU = np.zeros(num_labels, dtype=np.float32)
    best_prec1 = 100000
    train_style = "iid" 
    print "setting devices"
    gpus_ = args.gpu.replace(" ", "").split(',')
    if args.valfirst == 1:
        val_first = True
    else:
        val_first = False
    print "Training from epoch {}.".format(args.From)
    start_epoch = args.From
    gpus_temp = []
    for i in range(len(gpus_)):
        if len(gpus_[i])>0:
            gpus_temp.append(int(gpus_[i]))
    if len(gpus_temp) ==0:
        raise(RuntimeError("Illegal GPUS specified"))
    else:
        print 'Using devices {}'.format(gpus_temp)
    deviceids = gpus_temp
    modelname=args.basemodel
    dims = args.dims
    print "Setting input image size to {}.".format(dims) 
    print "Predicting on {} labels.".format(args.labels)
    print "setting training sequence length"
    sequence = int(args.sequence)
    sel_type = ("set", sequence)
    print "Training with sequence types: {}".format(sel_type)
    basemodel, basefile, save_name, best_name, train_style = getModelParams(args, deviceids)
    
    criterion = CrossEntropy2d
    criterion2 = nn.SmoothL1Loss().cuda(deviceids[0])

    model = basemodel
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                momentum=0.9,
                                weight_decay=0.00001)



    cudnn.benchmark = True
    trainroot = ''
    valroot = ''
    dataset = args.which
    print "Training on {}".format(dataset)
    if dataset == 'kitti':
        valdir =  '/data/hinthorn/workspace_hinthorn/exp/sequence/lists/val'
        traindir =  '/data/hinthorn/workspace_hinthorn/exp/sequence/lists/train'
    elif dataset == 'mapcar':
        valdir = '/data/hinthorn/workspace_hinthorn/exp/sequence/lists/mapcarVal'
        traindir = '/data/hinthorn/workspace_hinthorn/exp/sequence/lists/mapcarTrain'
    elif dataset == 'cityscape':
        valdir =  '/data/hinthorn/workspace_hinthorn/exp/cityscape/cityscape/list/pyval/'
        traindir =  '/data/hinthorn/workspace_hinthorn/exp/cityscape/cityscape/list/pytrain'
    elif dataset == 'cityscapeMulti':
        valdir =  '/data/hinthorn/workspace_hinthorn/exp/cityscape/cityscape/list/pyvalMulti/'
        traindir =  '/data/hinthorn/workspace_hinthorn/exp/cityscape/cityscape/list/pytrainMulti/'
    else:
        raise(RuntimeError("Not a legal dataset"))
#    valdir = traindir # For now, don't validate 
    print 'loading datasets...'

    train_loader = torch.utils.data.DataLoader(
        RNNLoader(traindir, trainroot, aug_type=1, tosize=dims, lbsize=dims,ignore_label=-1, lb_type='mixed', rand_seq_len = sel_type, shuffleInput=True, mode='train'),#("set", 20)),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        RNNLoader(valdir, valroot, aug_type=2, tosize=dims, lbsize=dims, ignore_label=-1, lb_type='mixed', rand_seq_len =sel_type_val, shuffleInput=False, mode='val'), 
        batch_size=1, shuffle=False,
        num_workers=workers, pin_memory=True)

    if val_first:
        print "Validating"
        loss, IOU, PREC, REC = validate(val_loader, model, criterion,train_style=train_style,epoch=-1)

        # remember best prec@1 and save checkpoint
        for j in range(len(IOU)):
            print("Label {}:\n\tIOU:\t{}\n\tPrecision:\t{}\n\tRecall:"
            "\t{}\n\tDiff from BTD:\t{}").format(j, IOU[j], PREC[j], REC[j], bestIOU[j] - IOU[j])


    print 'start training...'
    for epoch in xrange(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, criterion2, optimizer,train_style = train_style, epoch=epoch)

        print 'epoch', epoch, 'trained, starting validate...'
        # evaluate on validation set
        loss, IOU, PREC, REC = validate(val_loader, model, criterion,train_style=train_style,epoch=epoch)

        # remember best prec@1 and save checkpoint
        is_best = loss < best_prec1
        for j in range(len(IOU)):
            print "Label {}:\n\tIOU:\t{}\n\tPrecision:\t{}\n\tRecall:\t{}\n\tDiff from BTD:\t{}".format(j, IOU[j], PREC[j], REC[j], bestIOU[1] - IOU[1])
        if is_best:
            print "New PR: {}".format(loss)
        best_iou = IOU[1] > bestIOU[1]

        if best_iou:
            print "New IOU PR: {}".format(IOU)
            bestIOU = IOU
        best_prec1 = min(loss, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'bestIOU' : bestIOU,
            'optimizer' : optimizer.state_dict(),
        }, best_iou, filename=save_name)


def train(train_loader, model, criterion, criterion2, optimizer, train_style, epoch):
    global dims
    seq_time = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageWindow(length=(print_freq//calc_freq),init=0.01)
    losses = AverageWindow(length=(print_freq//calc_freq), init=3000)
    LOSS1 = AverageWindow(length=(print_freq//calc_freq), init=7000.0)
    batch_time = AverageMeter()
    IOUROAD = AverageMeter()
    I = AverageMeter()
    U = AverageMeter()
    IOU = AverageWindow(length=(print_freq//calc_freq), init=0.5)
    PREC = AverageWindow(length=(print_freq//calc_freq), init=0.5)
    REC = AverageWindow(length=(print_freq//calc_freq), init=0.5)
    torch.cuda.device(deviceids)
    model.train()

    l1 = 1.0 
    l2 = 0.15
    l3 = 0.15
    lp = 0.2

#Will:  Below is a torch code snippet that is supposed to turn of batch normalization layers. Won't use for now but could adapt if necessary    
#    model.apply(function(m) if torch.type(m).find("BatchNormalization") then m.evaluate():!)

    end = time.time()
    count = 0
    accum = 0
    print "Training an {} model".format(train_style)
    alpha = 2.0
    for k in range(batch_size):
        #for i, (imgnames, targnames) in enumerate(train_loader):
        for i, (imgnames, targnames, scale, hval, wval, reflect) in enumerate(train_loader):
            h, out2 = None, None
            h = model.module.init_hidden().cuda(deviceids[0], async=True)

            for (img, target, j) in zip(imgnames, targnames, range(len(targnames))): 
                #print "Image {} of length {}".format(j, len(targnames))
                #input, target, targetp = train_loader.dataset.preprocessImage(img, target, 0)
                input, target, targetp = train_loader.dataset.preprocessImage(img, target, 0, (scale, hval, wval, reflect))
                input = input.expand(1, 1, dims, dims).contiguous()
                data_time.update(time.time() - end)
                input_var = Variable(input.cuda(deviceids[0], async=True))
                target_var = Variable(target.cuda(deviceids[0], async=True))
                target_p = Variable(targetp.cuda(deviceids[0], async=True))
                out = None
                if train_style == 'kalman':
                    startTime = time.time()
                    
#                    print "Outside {} {}: {}".format(k, j, h.max())
                    #out, h, outparam = model(input_var, h, j) #input hidden layer            
                    out, h = model(input_var, h, j) #input hidden layer            

                    #print "Hidden size: {}".format(h.size())
                    forward_time.update(time.time() - startTime)
                    h = Variable(h.data) # Prevent BPTT from aggregating over multiple iterations
                    accum += 1
                else:
                    raise(RuntimeError("Train_style {} not recognized.".format(train_style)))
                loss1 = criterion(out, target_var, deviceids=deviceids)
#                lossp = criterion2(outparam, target_p)
#                lossp = lossp * 448
#                print "Lossp: ({})".format(lossp.data[0])
                LOSS1.update(loss1.data[0])
                if train_style == 'kalman': 
                    loss1 = l1 * loss1 / float(accum_batch)# average  sequence
                    print 'expanded {}'.format(loss1.data[0])
                    accum+=1
                losses.update(loss1.data[0]) #, input.size(0))
                loss1.backward(retain_variables=True)
#                lossp.backward(retain_variables=True)
                count = count + 1
                # Calculate average over window (Approximation)
                if count % calc_freq == 0:
                    iou, prec, rec, inters, union= jaccard(out, target_var)
                    I.update(inters)
                    U.update(union)

#                    print "Jaccard IOU: {}".format(iou)
                    if iou[1] >=0:
                        IOUROAD.update(iou)
                        IOU.update(iou)
                        PREC.update(prec)
                        REC.update(prec)
                    if save_training and epoch > min_save: # and epoch > 0: #Let it get through an epoch first.
                        imname=img[0].split('/')[-1][:-4] #Get name
                        imname = imname.replace("splitFlag_", "")
                        if len(imname) > 50:
                            head = imname[:30]
                            imname = head + imname[-20:]
                        imname = imname + "_" + str(iou[1]) + '_'
                        if iou[1] > 0.9:
                            print "{} > 0.90, saving\t({})".format(iou[1], loss1.data[0])
                            saveoutput(input, out, target_var, epoch, savename=imname, q='TRAIN_GOOD')
                        if iou[1] < 0.7:
                            print "{} < 0.90, saving\t({})".format(iou[1], loss1.data[0])
                            saveoutput(input, out, target_var, epoch, savename=imname, q='TRAIN_AWFUL')


                if count % print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           epoch, i, len(train_loader), batch_time=batch_time,
                           data_time=data_time, loss=losses))
                    print('Forward time:\t{}').format(forward_time.avg)
#                        iou, prec, rec = jaccard(out, target_var)
                    print('Average road IOU overall:\t{}').format(IOUROAD.avg)
                    print('New Calc Style:\t{}'.format(I.sum/U.sum))
                    print('Proportion: {}'.format(U.sum[0]/U.sum[1])
                    print('AVG Training Stats for round {}:\n\tIOU:'
                           '\t{}\n\tPREC:\t{}\n\tREC:\t{}'.format(count, 
                           IOU.avg, PREC.avg, REC.avg))
                    print('Out layer stats:\n\tMax:\t{}\n\tMin:\t{}'
                           '\n\tMean:\t{}\n\tStd:\t{}'.format(out.max().data[0],
                           out.min().data[0], out.mean().data[0], out.std().data[0]))

                if gradient_clip > 0: #Should I clip for all steps?
                    nn.utils.clip_grad_norm(model.parameters(), gradient_clip)
            
            # Update parameters after sequence
            if (train_style == 'kalman' or train_style == 'iid') and ((batch_size*accum) >= accum_batch):
                #print "Loss: {}".format(losses.avg*accum)
                optimizer.step()
                optimizer.zero_grad()
                batch_time.update(time.time() - end)
                end = time.time()
                accum = 0

def printStats(pred):
    print "Stats:\t{}\t{}".format(pred.data.max(), pred.data.min())


def validate(val_loader, model, criterion, train_style, epoch, num_labels=2):
    global dims
    batch_time = AverageMeter()
    losses = AverageMeter()
    jac = AverageMeter()
    IOUROAD = AverageMeter()
    IOU = np.zeros(num_labels, dtype=np.float32)
    I = AverageMeter()
    U = AverageMeter()
    REC = np.zeros(num_labels, dtype=np.float32)
    PREC = np.zeros(num_labels, dtype=np.float32)
    model.eval()
    end = time.time()
    count = np.zeros(num_labels, dtype=np.int)
    
    #for i, (imgnames, targnames) in enumerate(val_loader):
    for i, (imgnames, targnames, scale, hval, wval, reflect) in enumerate(val_loader):
        h = None
        if train_style == 'kalman' or train_style == 'rnnDual':
            h = model.module.init_hidden().cuda(deviceids[0], async=True)
        for (img, target, j) in zip(imgnames, targnames, range(len(targnames))): 
            #input, target, targetp = val_loader.dataset.preprocessImage(img, target, 0)
            input, target, targetp = val_loader.dataset.preprocessImage(img, target, 0, (scale, hval, wval, reflect))
            input = input.expand(1, 1, dims, dims).contiguous()
            input = input.cuda(deviceids[0], async=True)
            target = target.cuda(deviceids[0], async=True)
            targetp = targetp.cuda(deviceids[0], async=True)
            input_var = Variable(input, volatile=True)
            target_var = Variable(target, volatile=True)
            target_p = Variable(targetp, volatile=True)
            
            outp = None
            if train_style == 'kalman':
                #outp, h, outparam = model(input_var, h, j)
                outp, h= model(input_var, h, j)

            else:
                print "Training style undefined"
                exit()
            loss = criterion(outp, target_var)
#            lossp = criterion2(outparam, targetp)
            iou, prec, rec, inters, union= jaccard(outp, target_var)
            I.update(inters)
            U.update(union)
            if iou[1] >=0:
                IOUROAD.update(iou)
            for k in range(num_labels):
                if iou[k] >=0:
                    IOU[k] = IOU[k] + iou[k]
                    count[k] = count[k] + 1
            PREC = PREC + prec
            REC = REC + rec
            if save_validation and epoch > min_save:
                imname=img[0].split('/')[-1][:-4] #Get name
                imname = imname.replace("splitFlag_", "")
                if len(imname) > 30:
                    head = imname[:30]
                    imname = head + imname[-10:]
                imname = imname + "_" + str(iou[1]) + '_'
                if iou[1] > 0.95:
                    print "{} > 0.95, saving".format(iou[1])
                    saveoutput(input.cpu(), outp, target_var, epoch, savename=imname, q='VALGOOD')
                elif iou[1] < 0.8:
                    print "{} < 0.8, saving error".format(iou[1])
                    saveoutput(input.cpu(), outp, target_var, epoch, savename=imname, q='VALAWFUL')
                else:
                    saveoutput(input.cpu(), outp, target_var, epoch, savename=imname, q='VALOK')

            # measure accuracy and record loss
            losses.update(loss.data[0], input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if j % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses))
                print('Average road IOU overall:\t{}').format(IOUROAD.avg)
                print('New Calc Style:\t{}'.format(I.sum/U.sum))

    print 'Count:\t{}'.format(count)
    print "IOUROAD: {}".format(IOUROAD.avg)
    return losses.avg, I.sum/U.sum, PREC/count, REC/count #IOU/count, PREC/count, REC/count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print 'saving to ', filename
    if is_best:
        print 'saving best...'
        shutil.copyfile(filename, best_name)


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

def saveoutput(im, pred, targ, index, savename = '', q='AWFUL'):
    global dims
    root = '/data/hinthorn/workspace_hinthorn/exp/pytorch'
    savename = os.path.join(root, 'imgs',modelname +'_'+ savename+'_epoch_'+ str(index)+'_'+q + '.png')
#    print "Saving: {}".format(savename)
    pred = pred.data.cpu().squeeze().numpy()
    target = targ.data.cpu().squeeze().numpy()
    im = im.squeeze().numpy()
    im = (im - np.min(im))
    im = (im / np.max(im)) * 150
#    im = ((im+1)*200).squeeze().numpy()
    #print "Max image: {}, {}".format(np.max(im), np.min(im))
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = base_lr * (0.9 ** (epoch // 1))
    #log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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



if __name__ == '__main__':
    main()
