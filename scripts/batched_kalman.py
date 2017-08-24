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
from losses import *
from training_utils import *
#from CrossEntropy2d import CrossEntropy2d
from xentrpy2d import CrossEntropy2d 
from res18 import *
from res50_dilated import *
from res50_recurrent import *
from fetch_model import getModelParams
print_freq = 10
calc_freq = 1 
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
save_validation = True 
min_save = 2 
save_training =  False #True 
val_first = False 
accum_batch = -1
dims = 448
dataset = 'mapcar'
def main():
    global start_epoch, args, save_name, best_name, num_labels, modelname, deviceids, val_first, dims, dataset, accum_batch

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
    accum_batch = 8*sel_type[1]
    if accum_batch >30:
        accum_batch = 30
    print "Accumulating gradient over 8*{}={} frames".format(sel_type[1],8*sel_type[1])
    basemodel, basefile, save_name, best_name, train_style = getModelParams(args, deviceids)
    
    criterion = CrossEntropy2d
    criterion2 = nn.SmoothL1Loss().cuda(deviceids[0])

    model = basemodel
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                momentum=0.9,
                                weight_decay=0.0001)



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
        RNNLoader(valdir, valroot, aug_type=2, tosize=dims, lbsize=dims, ignore_label=-1, lb_type='mixed', rand_seq_len =sel_type, shuffleInput=False, mode='val'), 
        batch_size=1, shuffle=False,
        num_workers=workers, pin_memory=True)
    #val_loader = torch.utils.data.DataLoader(
    #    RNNLoader(valdir, valroot, aug_type=2, tosize=dims, lbsize=dims, ignore_label=-1, lb_type='mixed', rand_seq_len =sel_type_val, shuffleInput=False, mode='val'), 
    #    batch_size=1, shuffle=False,
    #    num_workers=workers, pin_memory=True)

    if val_first:
        print "Validating"
        loss, IOU, PREC, REC = validate(val_loader, model, criterion,train_style=train_style,epoch=start_epoch)

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
    forward_time = AverageWindow(length=((5*print_freq)//calc_freq),init=0.01)
    losses = AverageWindow(length=((5*print_freq)//calc_freq), init=3000)
    LOSS1 = AverageWindow(length=((5*print_freq)//calc_freq), init=7000.0)
    batch_time = AverageMeter()
    I = AverageMeter()
    U = AverageMeter()
    I2 = AverageMeter()
    U2 = AverageMeter()
    IOU = AverageWindow(length=((5*print_freq)//calc_freq), init=0.5)
    PREC = AverageWindow(length=((5*print_freq)//calc_freq), init=0.5)
    REC = AverageWindow(length=((5*print_freq)//calc_freq), init=0.5)
    IOU2 = AverageWindow(length=((5*print_freq)//calc_freq), init=0.5)
    PREC2 = AverageWindow(length=((5*print_freq)//calc_freq), init=0.5)
    REC2 = AverageWindow(length=((5*print_freq)//calc_freq), init=0.5)
    torch.cuda.device(deviceids)
    model.train()


    end = time.time()
    accum = 0
    print "Training an {} model".format(train_style)
    for k in range(batch_size):
        print "K={}".format(k)
        for i, (imgnames, targnames, scale, hval, wval, reflect) in enumerate(train_loader):
            out2 = None, None
            inputs, toPaint, targets = [],[],[]
            SEQLENGTH = len(targnames)

            #Collect all images in the sequence
            for (img, target, j) in zip(imgnames, targnames, range(SEQLENGTH)): 
                input, target, targetp = train_loader.dataset.preprocessImage(img, target, 0, (scale, hval, wval, reflect))
                input = input.expand(1, 1, dims, dims).contiguous()
                target = target.expand(1, 1, dims, dims).contiguous()
                data_time.update(time.time() - end)
                input_var = Variable(input.cuda(deviceids[0], async=True))
                target_var = Variable(target.cuda(deviceids[0], async=True))
                #target_p = Variable(targetp.cuda(deviceids[0], async=True))
                targets.append(target_var)
                inputs.append(input_var)
                toPaint.append(input)

            #Train by seqence
            input_var = torch.cat(inputs,0)
            target_var = torch.cat(targets,0)
            out = None
            if train_style == 'kalman':
                startTime = time.time()
                #out, out2, out3 = model(input_var) #input hidden layer  
                out= model(input_var) #input hidden layer  
                forward_time.update(time.time() - startTime)
                accum += SEQLENGTH 
            else:
                raise(RuntimeError("Train_style {} not recognized.".format(train_style)))


            loss1 = criterion(out, target_var, deviceids=deviceids)
            loss2 = criterion(out2, target_var, deviceids=deviceids)
            loss3 = criterion(out3, target_var, deviceids=deviceids)
            LOSS1.update(loss1.data[0])
            loss1 = loss1 / float(accum_batch)
            loss2 = loss2 / float(accum_batch)
            loss3 = loss3 / float(accum_batch)

            losses.update(loss1.data[0]) #, input.size(0))
            loss1.backward(retain_variables=True)
            loss2.backward(retain_variables=True)
            loss3.backward(retain_variables=True)

            # Calculate average over window (Approximation)
            toCheck = [0,1]
            for v in toCheck:
                b,c,s,_ = out.size()
                guess, right = out[v], target_var[v]
                guess = guess.expand(1,c,s,s)
                right = right.expand(1,1,s,s)
                iou, prec, rec, inters, union= jaccard(guess,right)
                I.update(inters)
                U.update(union)

                if iou[1] >=0:
                    IOU.update(iou)
                    PREC.update(prec)
                    REC.update(prec)
                guess = out2[v].expand(1,c,s,s)
                iou, prec, rec, inters, union= jaccard(guess,right)
                I2.update(inters)
                U2.update(union)

                if iou[1] >=0:
                    IOU2.update(iou)
                    PREC2.update(prec)
                    REC2.update(prec)

                if save_training and epoch > min_save:
                    imname=imgnames[v][0].split('/')[-1][:-4] #Get name
                    if len(imname) > 30:
                        head = imname[:11]
                        st = imname.find('drive')
                        if st > 0:
                            imname = imname[st:]
                        imname = imname.replace("splitFlag_","")

                        imname = imname.replace("sync_","")
                        imname = imname.replace("data_","")
                        imname = head + imname
                    imname = imname + '_' + str(v) + "_" + str(iou[1]) + '_'
                    print 'saving {}'.format(imname)
                    q = "NONE"
                    if iou[1] > 0.95:
                        q = 'VALGOOD'
                    elif iou[1] < 0.8:
                        q = 'VALAWFUL'
                    else:
                        q = 'VALOK'
                   # print "{}: {}".format(q, iou[1])
#                    saveoutput(input_var.data[v].cpu(), out[v], target_var[v], dims=dims, epoch=epoch, savename=imname, q=q, modelname=modelname)
                    savefuzzyoutput(input_var.data[v].cpu(), out[v], target_var[v],  epoch, savename=imname, q=q, modelname=modelname)


            if (i) % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses))
                print('Forward time:\t{}').format(forward_time.avg)
                print('Entr IOU Overall:\t{}'.format(I.sum/U.sum))
                print('Cont IOU Overall:\t{}'.format(I2.sum/U2.sum))
                print('AVG Training Stats for round {}:\n\tIOU:'
                       '\t{}\n\tPREC:\t{}\n\tREC:\t{}'.format(i*j, 
                       IOU.avg, PREC.avg, REC.avg))
                print('AVG Training Stats for round {}:\n\tIOU:'
                       '\t{}\n\tPREC:\t{}\n\tREC:\t{}'.format(i*j, 
                       IOU2.avg, PREC2.avg, REC2.avg))
                print('Out layer stats:\n\tMax:\t{}\n\tMin:\t{}'
                       '\n\tMean:\t{}\n\tStd:\t{}'.format(out.max().data[0],
                       out.min().data[0], out.mean().data[0], out.std().data[0]))

            if gradient_clip > 0: #Should I clip for all steps?
                nn.utils.clip_grad_norm(model.parameters(), gradient_clip)
            
            # Update parameters after sequence
            if (train_style == 'kalman' or train_style == 'iid') and (accum >= accum_batch):
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
    IOU = np.zeros(num_labels, dtype=np.float32)
    IOU_ = AverageMeter()
    REC = AverageMeter() 
    PREC = AverageMeter() 
    I = AverageMeter()
    U = AverageMeter()


    I2 = AverageMeter()
    U2 = AverageMeter()
    REC2 = AverageMeter() 
    PREC2 = AverageMeter() 
    IOU_2 = AverageMeter()
    model.eval()
    end = time.time()
    
    #for i, (imgnames, targnames) in enumerate(val_loader):
    for i, (imgnames, targnames, scale, hval, wval, reflect) in enumerate(val_loader):
            inputs, toPaint, targets = [],[],[]
            #Collect all images in the sequence
            for (img, target, j) in zip(imgnames, targnames, range(len(targnames))): 
                input, target, targetp = val_loader.dataset.preprocessImage(img, target, 0, (scale, hval, wval, reflect))
                input = input.expand(1, 1, dims, dims).contiguous()
                target = target.expand(1, 1, dims, dims).contiguous()

                input_var = Variable(input.cuda(deviceids[0], async=True))
                target_var = Variable(target.cuda(deviceids[0], async=True))
                #target_p = Variable(targetp.cuda(deviceids[0], async=True))
                targets.append(target_var)
                inputs.append(input_var)
                toPaint.append(input)



            input_var = torch.cat(inputs,0)
            target_var = torch.cat(targets,0)
            #out, out2, _ =  model(input_var)
            out =  model(input_var)
            loss = criterion(out, target_var)
            b,c,s,_ = out.size()
            for v in range(b):
                iou, prec, rec, inters, union= jaccard(out[v].expand(1,c,s,s), target_var[v].expand(1,1,s,s))
                I.update(inters)
                U.update(union)
                for k in range(num_labels):
                    if iou[k] >=0:
                        IOU[k] =  iou[k]
                IOU_.update(IOU)
                PREC.update(prec)
                REC.update(rec)
            for v in range(b):
                iou, prec, rec, inters, union= jaccard(out2[v].expand(1,c,s,s), target_var[v].expand(1,1,s,s))
                I2.update(inters)
                U2.update(union)
                for k in range(num_labels):
                    if iou[k] >=0:
                        IOU[k] =  iou[k]
                IOU_2.update(IOU)
                PREC2.update(prec)
                REC2.update(rec)

                if save_validation and epoch > min_save:
                    imname=imgnames[v][0].split('/')[-1][:-4] #Get name
                    if len(imname) > 30:
                        head = imname[:11]
                        st = imname.find('drive')
                        if st > 0:
                            imname = imname[st:]
                        imname = imname.replace("splitFlag_","")
                        imname = imname.replace("sync_","")
                        imname = imname.replace("data_","")
                        imname = head + imname
                    imname = imname + '_' + str(v) + "_" + str(iou[1]) + '_'
                    q = "NONE"
                    if iou[1] > 0.95:
                        print "{} > 0.95, saving".format(iou[1])
                        q = 'VALGOOD'
                    elif iou[1] < 0.8:
                        print "{} < 0.8, saving error".format(iou[1])
                        q = 'VALAWFUL'
                    else:
                        q = 'VALOK'
                    #saveoutput(input_var.data[v].cpu(), out[v], target_var[v],dims=dims,  epoch=epoch, savename=imname, q=q, modelname=modelname)
                    savefuzzyoutput(input_var.data[v].cpu(), out[v], target_var[v],  epoch, dims=dims, savename=imname, q=q, modelname=modelname)

            # measure accuracy and record loss
            losses.update(loss.data[0], input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses))
                print('Entrp Prec Rec:\t{}\t{}'.format(PREC.avg, REC.avg))
                print('Contr Prec Rec:\t{}\t{}'.format(PREC2.avg, REC2.avg))
                print('Entrp IOU:\t{}'.format(I.sum/U.sum))
                print('Control IOU:\t{}'.format(I2.sum/U2.sum))

    return losses.avg, I.sum/U.sum, PREC.avg, REC.avg 


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print 'saving to ', filename
    if is_best:
        print 'saving best...'
        shutil.copyfile(filename, best_name)








def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = base_lr * (0.9 ** (epoch // 1))
    #log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr






if __name__ == '__main__':
    main()
