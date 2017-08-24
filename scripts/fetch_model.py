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
from tiramisu import *
from kalman2 import *
from kalmanlong import *
from entropygates import *






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
    elif args.basemodel == 'Tiramisu40':
        ID = 1 
        sw = 0 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basefile = 'models/'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = Tiramisu40(args.labels, args.dims, 0, deviceids=deviceids[0])
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"
    elif args.basemodel == 'TiramisuRec40':
        ID = 1 
        sw = 0  
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = Tiramisu40(args.labels, args.dims, 1, deviceids=deviceids[0])
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"
    elif args.basemodel == 'TiramisuRecMultOut40':
        ID = 1 
        sw = 0 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = Tiramisu40(args.labels, args.dims, 2, deviceids=deviceids[0])
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"
    elif args.basemodel == 'TiramisuRecMultIn40':
        ID = 1 
        sw = 0 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = Tiramisu40(args.labels, args.dims, 3, deviceids=deviceids[0])
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"
    elif args.basemodel == 'TiramisuRecMultBoth40':
        ID = 1 
        sw = 0 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = Tiramisu40(args.labels, args.dims, 4, deviceids=deviceids[0])
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"
#    elif args.basemodel == 'TiramisuRecAddSub40':
#        ID = 0 
#        sw = 0 
#        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
#        basemodel = Tiramisu40(args.labels, args.dims, 5)
#        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
#        best_name = 'models/best_'+save_name
#        save_name = 'models/' + save_name
#        train_style = "kalman"
    elif args.basemodel == 'TiramisuRecSubSub40':
        ID = 1 
        sw = 0 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = Tiramisu40(args.labels, args.dims, 6, deviceids=deviceids[0])
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"
    elif args.basemodel == 'TiramisuRecSubMult40':
        ID = 0 
        sw = 0 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = Tiramisu40(args.labels, args.dims, 7, deviceids=deviceids[0])
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"
    elif args.basemodel == 'TiramisuRecSubMultGatedELU40':
        ID = 0 
        sw = 0 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = Tiramisu40(args.labels, args.dims, 8, deviceids=deviceids[0])
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"
    elif args.basemodel == 'TiramisuRecSubMultGatedID40':
        ID = 0 
        sw = 0 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = Tiramisu40(args.labels, args.dims, 9, deviceids=deviceids[0])
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"
    elif args.basemodel == 'DenseShuffle23':
        ID = 5 
        sw = 5 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = DenseShuffle23(args.labels, args.dims)
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "iid"
    elif args.basemodel == 'DenseShuffle23Trans':
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
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = resrecDualGRU50(False)
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "rnnDual"






# Back to Bacis with the FCN:
    

    elif args.basemodel == 'PREDCORR':
        ID = 1
        sw = 1 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = FCNDUCLONG(args.labels,args.sequence,'kalman')
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"
    elif args.basemodel == 'PREDCORRA':
        ID = 1
        sw = 1 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
#        basefile = 'models/best_'+"PREDCORR"+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = FCNDUCLONG(args.labels,args.sequence,'kalmana')
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"
    elif args.basemodel == 'FCNDUCLONG':
        ID = 1
        sw = 1 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
       # basefile = 'models/best_'+'FCNDUCLONGT'+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = FCNDUCLONG(args.labels,args.sequence,'iid')
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"
    elif args.basemodel == 'FCNDUCLONGT':
        ID = 1
        sw = 1 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = FCNDUCLONG(args.labels,args.sequence,'iidT')
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"

    elif args.basemodel == 'EntropyGate':
        ID = 2
        sw = 1 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = EntropyNet(args.labels,args.sequence,'kalman', True)
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"
    elif args.basemodel == 'EntropyGateControl':
        ID = 0
        sw = 0 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = EntropyNet(args.labels,args.sequence,'kalman', False)
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"

    elif args.basemodel == 'FCNDUCL':
        ID = 1
        sw = 1 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = FCNDUCLONG(args.labels,args.sequence,'iid')
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"
    elif args.basemodel == 'PREDCORRL':
        ID = 1
        sw = 1 
        basefile = 'models/best_'+args.basemodel+'-checkpoint_ID'+str(ID-sw)+'.pth.tar'
        basemodel = FCNDUCLONG(args.labels,args.sequence,'kalman')
        save_name = args.basemodel+'-checkpoint_ID'+str(ID)+'.pth.tar'
        best_name = 'models/best_'+save_name
        save_name = 'models/' + save_name
        train_style = "kalman"



    else:
        raise(RuntimeError("Illegal architecture name")) 

    
    #basemodel = torch.nn.DataParallel(basemodel, device_ids=deviceids).cuda(deviceids[0])
    if os.path.isfile(basefile):
        print "=> loading checkpoint '{}'".format(basefile)
        checkpoint = torch.load(basefile)
        model_dict = basemodel.state_dict()
        print "Model_dict: {}".format(len(model_dict))
        print "Checkpoint: {}".format(len(checkpoint['state_dict']))
        pretrained_dict = {k:v for k, v in checkpoint['state_dict'].items() if (k in model_dict)}# and model_dict[k].size() == v.size())}
        l = len(pretrained_dict)
        d = {}
        if l == 0:
            print "Length: {}. Attempting to shortened name".format(len(pretrained_dict))
            for k,v in checkpoint['state_dict'].items():
                k = k.replace('module.','')
                if k in model_dict:
                    if model_dict[k].size() == v.size():
                        d[k] = v
                    else:
                        print '{}: Size Incompatability.'.format(k)
                else:
                    print 'Still incompatible: {}'.format(k)
            pretrained_dict = d
        print "Length: {}".format(len(pretrained_dict))

        
        pretrained_dict = d
        model_dict.update(pretrained_dict)
        basemodel.load_state_dict(model_dict)
        state = checkpoint['optimizer']
        print "=> loaded checkpoint '{}'" \
              .format(basefile)
    else:
        print "{} not found. Training from scratch".format(basefile)
    basemodel = torch.nn.DataParallel(basemodel, device_ids=deviceids).cuda(deviceids[0])
    return basemodel, basefile, save_name, best_name, train_style 




