import numpy as np
import cv2
import time
import os
import sys
sys.path.append('./include')
sys.path.append('./ark')
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transform
from PIL import Image
from rnn_loader import RNNLoader

from losses import *
from CrossEntropy2d import CrossEntropy2d
#from recurrent_resnet import *
from res50_dilated import *
from res50_recurrent import *
show_imh=600
show_imw=960

ipm_h=600
ipm_w=300

lb_to_RGB = {
        0: [20, 20, 20],
        1: [0, 255, 0],
        2: [20, 50, 20],
        3: [50, 20, 50],
        4: [244, 35, 232],
        5: [230, 150, 140],
        6: [70, 100, 70],
        7: [190, 153, 153],
        8: [150, 120, 90],
        9: [150, 120, 150],
        10:[250, 170, 30],
        11: [220, 220, 0],
        12: [107, 142, 35],
        13: [152, 251, 152],
        14: [70, 130, 180],
        15: [220, 20, 60],
        16: [220, 100, 60],
        17: [0, 0, 142],
        18: [0, 0, 70],
        19: [0, 60, 100],
        20: [0, 0, 90],
        21: [0, 0, 110],
        22: [0, 80, 100],
        23: [0, 0, 230],
        24: [119, 11, 32],
        25: [0, 0, 142],
        26: [81, 0, 81]
        }


#RG_values = np.array(np.random.rand(num_labels, 2) * 10, dtype=np.uint8)



im_h=448
im_w=448
im_ch=1
im_mean=128
im_scale=1

out_h = 448
out_w = 448

parser = argparse.ArgumentParser(description='Model Testing and Cityscape Predictions')
parser.add_argument('--model', '-m', action="store", default='multi', type=str)
args = parser.parse_args()


model = args.model #'tidalWave'
save_photos = True
cwd = os.getcwd()

deviceids=[1]
workers=8
criterion = CrossEntropy2d
print "Testing model {}".format(model)
rootfolder= os.path.join(cwd, 'ark')
basefile = None
basemodel = None
train_style = None
if model == 'res50-DUC':
    ID=6
    basefile = 'models/res50-DILTID-checkpoint_ID'+str(ID)+'.pth.tar'
    basemodel = resnetDUC50(False)
    basemodel = torch.nn.DataParallel(basemodel, device_ids=deviceids).cuda(deviceids[0])
    train_style = 'iid'
elif model == 'resrecElm50':
    ID=1
    basefile = 'models/bestres50-REC-DILTID-checkpoint_ID'+str(ID)+'.pth.tar'
    basemodel = resrecElm50(True)
    basemodel = torch.nn.DataParallel(basemodel, device_ids=deviceids).cuda(deviceids[0])
    train_style = 'rnn'
else:
    print "{} is not a valid model. Aborting.".format(args.model)
    exit()
if os.path.isfile(basefile):
    print "=> loading checkpoint '{}'".format(basefile)
    checkpoint = torch.load(basefile)
    model_dict = basemodel.state_dict()
    pretrained_dict = {k:v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    basemodel.load_state_dict(model_dict)
    state = checkpoint['optimizer']
    print "=> loaded checkpoint '{}'" \
          .format(basefile)
model=basemodel
#    model = os.path.join(rootfolder, 'recurrent_train')
#    weights = os.path.join(rootfolder, 'models', 'round2_iter_30000.caffemodel')
save_folder = os.path.join(rootfolder,'results', args.model)
print "Saving to {}".format(save_folder)
num_labels = 2
out_ch = 2




testdir =  '/data/hinthorn/workspace_hinthorn/exp/sequence/lists/'
testroot = ''

input_blobname='data'
split_flag='||'
rootImFolder='/data/hinthorn/data/parsing/kitti_v1/select_imgs/'
rootLbFolder='/ssd/hinthorn/data/kitti_v1/data/'
batch_size=1

print "Creating test-loader"
test_loader = torch.utils.data.DataLoader(
        RNNLoader(testdir, testroot, aug_type=2, tosize=448, lbsize=448, ignore_label=-1, lb_type='map', rand_seq_len = ("full", 0)),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
print "Done. Setting model to eval()"
result_raw=[]
IOU = np.zeros(num_labels, dtype=np.float32)
I = np.zeros(num_labels, dtype=np.float32)
U = np.zeros(num_labels,dtype=np.float32)
REC = np.zeros(num_labels, dtype=np.float32)
PREC = np.zeros(num_labels, dtype=np.float32)
count = 0 # files
count_ = 0 # images
times = []
model.eval()
print "Done. Iterating"
for i, (imgs, lbs) in enumerate(test_loader):
    if count > 1: # Only check a couple sequences (for time)
        print "Leaving loop"
        break
    print "Round: {}\nImgs:\t{}\nLbs:\t{}\n".format(i, len(imgs), len(lbs))
    h = None
    if train_style == 'rnn':
        h = model.module.init_hidden().cuda(deviceids[0], async=True)
    for (img, lb, j) in zip(imgs, lbs, range(len(lbs))):
#        print "Processing image {}".format(j)
#        print "Length of files: {}, {}".format(len(imgs), len(lbs))
        lb_ = lb
        im, lb=test_loader.dataset.preprocessImage(img, lb, 0)
        showim, _=test_loader.dataset.preprocessImage(img, lb_, 0, encoding='RGB')
        input = im.expand(1, 1, 448, 448).contiguous()
        input_var = torch.autograd.Variable(input.cuda(deviceids[0], async=True))
        target_var = torch.autograd.Variable(lb.cuda(deviceids[0], async=True))
        begin = time.clock()
        input_var = input_var.contiguous()
        if train_style == 'iid':
            predictmap = model(input_var)#, h) #input hidden layer
        elif train_style == 'rnn':
            predictmap, h = model(input_var, h)
            h = Variable(h.data)
        times.append(time.clock() - begin)
        values, predictmap = predictmap.max(1)
        predictmap = predictmap.cpu().data.byte().squeeze().numpy()
        predictmap = cv2.resize(predictmap, (show_imw, show_imh), interpolation=cv2.INTER_LINEAR)
        resultid = img[0]
        showim = showim.numpy().squeeze()

	if len(showim.shape) == 2:
            showim = cv2.resize(showim,(show_imw,show_imh),interpolation=cv2.INTER_LINEAR)
            temp = np.zeros([showim.shape[0],showim.shape[1],3], dtype=np.uint8)
	    for c in range(0,3):
	        temp[:,:,c]=showim
	    showim = temp
        else:
           temp = np.zeros([show_imh, show_imw, 3], dtype=np.uint8)
           for c in range(0, 3):
               t = showim[:,:,c].copy()
#               print "min, max, mean: {}, {}, {}".format(np.min(t), np.max(t), np.mean(t))
               t = t - np.min(t)
               t = t * 255
               t = np.array(t, dtype=np.uint8)
               s = cv2.resize(t,(show_imw,show_imh),interpolation=cv2.INTER_LINEAR)
               temp[:, :, c]= cv2.resize(t,(show_imw,show_imh),interpolation=cv2.INTER_LINEAR)
           showim = temp

        lab = lb.byte().numpy()
        lab = cv2.resize(lab, (show_imw, show_imh), interpolation=cv2.INTER_LINEAR)
        for j in range(num_labels):
            I[j] = np.sum(np.logical_and((predictmap == j),(lab == j)))
            U[j] = np.sum(np.logical_or(np.logical_and(predictmap == j, lab != 255), lab == j)) 
            PREC[j] = PREC[j] + (I[j]/(np.sum(predictmap == j)+0.000001))
            REC[j] = REC[j] + (I[j]/(np.sum(lab == j)+0.000001)) #smoothing

            #ignore appropriate pixels
            if U[j] == 0: #model got all of it...
                I[j] = 1
                U[j] = 1
        iou = I/(U + 0.000001)
        print "this round i/u = {0}".format(iou)
        IOU = IOU + iou
        
        if save_photos == True: 
       #     print "Saving Photos"
            #color incorrect red
            showim = np.array(showim, dtype=np.float32)
            mask = np.zeros((showim.shape), dtype=np.float32) #Don't mask anything..
            #Fill in colors
            for i in range(num_labels):
                idx = predictmap == i
                c = lb_to_RGB[i]
#                print "ask, idx, showim:\t{}, {},{}".format(mask.shape, idx.shape, showim.shape)
                mask[idx, 0] = c[0]
                mask[idx, 1] = c[1]
                mask[idx, 2] = c[2]

        


            realroad = lab == 1
            mask[realroad, 0] = 100 # paint it red
            alpha=0.3
            showim = showim*(1-alpha) + mask*alpha #+ predictmap*beta
            result = showim
#            print "Name:\t{}".format(resultid)
            resultid = resultid.split('/')[-1]
            savename = os.path.join(save_folder,resultid)
            cv2.imwrite(savename,result)
            result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
            result_raw.append(result)
#            print "Image {} Done: {}".format(count, savename)
            count_ = count_ + 1
    count = count + 1

print "IOU SCORES  SUMMARY FOR {0}:".format(model)
IOU = IOU / count_
PREC = PREC / count_
REC = REC / count_
for i in range(num_labels):
    print "Class {0}:\n\tIOU: {1}\n\tPrecision: {2}\n\tRecall: {3}".format(i, IOU[i], PREC[i], REC[i])

print "Unweighted Averages:\n\tIOU: {0}\n\tPrecision: {1}\n\tRecall: {2}".format(np.mean(IOU), np.mean(PREC), np.mean(REC))
times = np.array(times)
print "Average time: {}".format(times.mean())
print "Mean fps: {}".format(1./times.mean())
if save_photos:
    savename = os.path.join(save_folder,'video'+'.mp4')
    track.encode_video(result_raw,savename)
    track.clear()
 
