import sys
import os
import torch
sys.path.append('./ark')
from res50_dilated import *
from res50_recurrent import *
from res18 import *
from shufflenet import *
from kalman import *
from leapfrog import *
from attention_net import *
from denseshuffle import *






def getModelParams(args, deviceids):
    basemodel, basefile, save_name, best_name, train_style = None, None, None, None, None
    print "Retrieving model {}".format(args.basemodel)
    if args.basemodel == 'resnetDUC50':
        basemodel = resnetDUC50(False)
        ID= 9 #Which run
        sw = 0#1
        #basefile = 'models/res50-DILTID-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        #basefile = 'models/bestres50-DILTID-checkpoint_ID6.pth.tar'
#        save_name = 'res50-DILTID-checkpoint_ID'+str(ID)+'.pth.tar'
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = 'iid'
    elif args.basemodel == 'resrecElm50':
        ID = 3
        sw = 1 
        basefile = 'models/bestres50-REC-DILTID-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = resrecElm50(False)
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "rnn"
    elif args.basemodel == 'Kalman':
        ID = 0
        sw = 0
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = Kalman(args.labels, args.dims)
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"
    elif args.basemodel == 'KalmanRecurrent':
        ID = 1
        sw = 0
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = Kalman(args.labels, args.dims,mode="rec")
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"
    elif args.basemodel == 'ResNeXt18':
        ID = 4
        sw = 0
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = ResNeXt18(False)
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "iid"
    elif args.basemodel == 'ResNeXtHD18':
        ID = 1
        sw = 0
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basefile = 'models/'+'best_ResNeXt18-checkpoint_ID2.pth.tar'
        
        basemodel = ResNeXt18(False, 'HD', insize=args.dims)
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "iid"
    elif args.basemodel == 'ResNeXtDense18':
        ID = 4
        sw = 1 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = ResNeXt18(False, 'regdense')
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "iid"
    elif args.basemodel == 'ShuffleNet28':
        ID = 5 
        sw = -1 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = ShuffleNet28(args.labels, args.dims)
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "iid"
    elif args.basemodel == 'LeapNet':
        ID = 3 
        sw = 0 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = LeapNet7(args.labels, args.dims)
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "iid"
    elif args.basemodel == 'DenseShuffle23':
        ID = 0 
        sw = 0 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = DenseShuffle23(args.labels, args.dims)
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "iid"
    elif args.basemodel == 'ResGlimpse18':
        ID = 1
        sw = 0
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basefile = 'models/'+'best_ResNeXt18-checkpoint_ID2.pth.tar'
        
        basemodel = ResGlimpse18(False)
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "glimpse"
    elif args.basemodel == 'resrecDualElm50':
        ID = 5
        sw = 1 
      #  basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = resrecDualElm50(False)
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "rnnDual"
    elif args.basemodel == 'resrecGRU50':
        ID = 3
        sw = 1 
        basemodel = resrecGRU50(False)
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
     #   basefile = 'models/best.res50-REC-DILTID-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "rnn"
    elif args.basemodel == 'resrecDualGRU50':
        ID = 3
        sw = 1 
        #basefile = 'models/best_'+'resrecDualElm50'+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = resrecDualGRU50(False)
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "rnnDual"
    else:
        raise(RuntimeError("Illegal architecture name")) 

    
    basemodel = torch.nn.DataParallel(basemodel, device_ids=deviceids).cuda(deviceids[0])
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
    else:
        print "{} not found. Training from scratch".format(basefile)
    return basemodel, basefile, save_name, best_name, train_style 




