import sys
sys.path.append('./include')
sys.path.append('./ark')
from rnn_loader import RNNLoader 
# from include/CrossEntropy2d import CrossEntropy2d
import argparse
import shutil
import time
import os
import torch
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
from CrossEntropy2d import CrossEntropy2d
#from recurrent_resnet import *
from res50_dilated import *
from res50_recurrent import *
from fetch_model import getModelParams
print_freq = 100
batch_size = 1
workers = 8
epochs = 300 
deviceids = [4]
base_lr = 1e-2 
gradient_clip = 1
start_epoch = 0
sel_type = ("set", 5)
sel_type_val = ("set", 50)
#hard_boost = True
modelname = ''
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--resume', '-r', action="store", default=0, type=int)
parser.add_argument('--basemodel', '-b', action="store", default='resrecElm50', type=str)
parser.add_argument('--gpu', '-g', action="store", default="0", type=str)
parser.add_argument('--dims', '-d', action="store", default=448, type=int)
parser.add_argument('--labels', '-l', action='store', default=2, type=int)
parser.add_argument('--which', '-w', action='store', default='kitti', type=str)
save_validation = True
save_training = False
def main():
    global start_epoch,num_labels, args, save_name, best_name, num_labels, modelname, deviceids
    args = parser.parse_args()
    num_labels = args.labels
    bestIOU = np.zeros(num_labels, dtype=np.float32)
    best_prec1 = 100000
    train_style = "iid" 
    print "setting devices"
    gpus_ = args.gpu.replace(" ", "").split(',')
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
#    basemodel = res18.resnetxelu18(True)
    #basemodel, basefile, save_name, best_name, train_style = getModelParams(args)
    basemodel, basefile, save_name, best_name, train_style = getModelParams(args, deviceids)
    #basemodel = torch.nn.DataParallel(basemodel, device_ids=deviceids).cuda(deviceids[0])
    
    criterion = CrossEntropy2d
    criterion2 = nn.SmoothL1Loss().cuda(deviceids[0])

    model = basemodel
    
    print "Getting Parameters"
    parameters = model.state_dict()
    ofInterest = {}
    #queries = ['rnn', 'rnninit']
    #queries = ['layer4']
    queries = ['glob']
    exclude = ['.bn']
    for k,v in parameters.items():
        print k
        for q in queries: 
    #        print "(q, k): ({}, {})".format(q.lower(), k.lower())
            if q.lower() in k.lower() and k.lower() not in exclude:
                found = False
                for e in exclude:
                    if e.lower() in k.lower():
                        found = True
                        break
                if not found:
                    ofInterest[k] = v 


    weights = {}
    biases = {}
    for k, v in ofInterest.items():
        if 'weight' in k: 
            weights[k] = v 
        elif 'bias' in k:
            biases[k] = v
#    savename = "retina_weights.png" 
#    import cv2
#    from PIL import Image
#    im = weights['module.retina.weight'].cpu().squeeze().numpy() 
#    im = im - im.min()
#    im = im / im.max()
#    im = im * 250
#    image = Image.fromarray(np.uint8(im))
#    im = image.resize((540, 540), Image.NEAREST)
#    im = np.array(im) 
#    cv2.imwrite(savename, im)
    for k, v in weights.items():
        print "{}'s weights are :".format(k)

        print v.size(), v.mean(0).mean(1), v.min(), v.max(), v.std()
        print ""

#    weights = parameters['DUC']

    #print "Weights: {}".format(weights)
    #for m in model.children():
    #    print "Model: {}".format(m)



















if __name__ == '__main__':
    main()
