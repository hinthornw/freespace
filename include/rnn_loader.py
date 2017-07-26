import torch.utils.data as data
import torch
import numpy as np
import numpy.random as rand
from PIL import Image
import cv2
import os
import os.path
from random import shuffle

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def makeSeqDataset(txtdir, root): #Make this randomized
    allfilenames = []
    ext = '.txt'
    for dirs, dirnames, files in os.walk(txtdir):
            for f in files:
                e = os.path.splitext(f)[-1]
                if e.lower() == ext:
                    filename = os.path.abspath(os.path.join(dirs, f))
                    allfilenames.append(filename)
    # Pick random sequences
    shuffle(allfilenames) # Make the order of drives random
    
    all_images = []
    for txtdir in allfilenames:
        images=[]
        with open(txtdir) as f:
            for line in f:
                parts = line.rstrip().split('||')
                if is_image_file(parts[0]):
                    images.append((os.path.join(root, parts[0]), parts[1]))
            all_images.append(images)
    #shuffle(all_images)
    return all_images




class RNNLoader(data.Dataset):

    def __init__(self, txtdir, root, aug_type=0, transform=None, target_transform=None, ignore_label=255, tosize=224, lbsize=None, lb_type='map', rand_seq_len=("set", 20), shuffleInput=True, mode='train'):

        all_imgs = makeSeqDataset(txtdir, root) #WILL made randomzized
        if len(all_imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        
        if mode=='val':
            #all_imgs = all_imgs[:3] #hack to make sure we don't evaluate EVERYTHING
            pass
        elif mode=='train':
            pass
        elif mode=='test':
            pass
        else:
            raise(RuntimeError("Illegal loading mode.")) 
        tp = rand_seq_len[0].lower()   
        k = 1
        N = [len(i) for i in all_imgs] #Note all_imgs is a list of lists of tuples: [Videos [frames(img, label)]]
        
        if tp == 'random': #Random lengthed sequence [1, film length]
            #Could try restricting to smaller range...
            k = [np.random.randint(n-2)+2 for n in N] #Must be at least size 2
#            a = [np.random.randint(n-k[i]) for n, i in zip(N, range(len(N)))]
            over = [k[i] % n for n, i in zip(N, range(len(N)))]
            a = [np.random.randint(over[i]) for n, i in zip(N, range(len(N)))]
            imgs_ = []
            for i in range(len(all_imgs)):
                for j in range(a[i], N[i]-a[i], k[i]):
                        imgs_.append(all_imgs[i][j:j+k[i]])
                    #all_imgs = [all_imgs[i][a[i]:a[i]+k[i]] for i in range(len(a))]
            all_imgs = imgs_
            print 'Retrieved {} chunks'.format(len(all_imgs))
            all_imgs = [all_imgs[i][a[i]:a[i]+k[i]] for i in range(len(a))]
        elif tp == 'full': #Full sequence
            pass
        elif tp == 'set': #Group by a fixed sequence length
            k = [rand_seq_len[1] for n, i in zip(N, range(len(N)))] #k-> length of each sequence
            over = [k[i] % n for n, i in zip(N, range(len(N)))]     #over-> excess after divying
            a = [np.random.randint(over[i]) for n, i in zip(N, range(len(N)))]  #a->start point
            imgs_ = []
            for i in range(len(all_imgs)):
                for j in range(a[i], N[i]-a[i], k[i]):
                        imgs_.append(all_imgs[i][j:j+k[i]])
            all_imgs = imgs_
            print 'Retrieved {} chunks'.format(len(all_imgs))
            if shuffleInput:
                shuffle(all_imgs)
        else:
            raise(RuntimeError("Sequence Preferences: Legal values are random, full, and set"))
        
        
        self.root = root
        self.all_imgs = all_imgs
        self.aug_type = aug_type
        if aug_type > 0:
            self.scale, self.hval, self.wval = self.randSeqParams()


        self.transform = transform
        self.target_transform = target_transform
        self.size = tosize
        if lbsize is None:
            self.lbsize = tosize
        else:
            self.lbsize = lbsize
        self.ignore_label = ignore_label
        self.lb_type = lb_type

    def __getitem__(self, index): #TODO: ensure cropping is continuous over entire series
        imgs = self.all_imgs[index]
        impath, lbpath = [], []
        for im, path in imgs: #self.all_imgs[index]
#            impath.append(Image.open(im).convert('L'))
#            lbpath.append(Image.open(path))
            impath.append(im)
            lbpath.append(path)
        return impath, lbpath# still only the file names

    def preprocessImage(self, impath, lbpath, index, encoding='L'):
        img = Image.open(impath[index]).convert(encoding)
        lb = Image.open(lbpath[index])
        #img, lb = Image.open(impath).convert('L'), Image.open(lbpath) #'RGB'
#        print "Original Im|label size:\t{}|{}".format(img.size, lb.size)
        if self.aug_type > 0:
            img, lb = self.randCropSeq(img, lb) #, self.scale, self.hval, self.wval)
#            img, lb = self.randCropList(impath, lbpath)

        if self.lb_type != 'map':
            paramlb = self.getparam(lb) 
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            lb = self.target_transform(img)
        if self.lb_type == 'dist':
            return img, self.lb_to_dist(lb)
        if self.lb_type == 'map':
            return img, lb
        elif self.lb_type == 'param':
            return img, paramlb
        elif self.lb_type == 'mixed':
            return img, lb, paramlb

    def __len__(self):
        return len(self.all_imgs)
    

    def randSeqParams(self):
        ''' Returns parameters for randCropSeq()'''
        if self.aug_type == 1:
            #random scale
            scale = 0.9 + rand.random() * 0.2
            wval, hval = rand.random(), rand.random()
        else:
            scale = 1
            hval, wval = None, None
        return scale, hval, wval

    def randCropList(self, ims, labels):
        ims_, labels_ = [], []
        for im, label in zip(ims, labels):
            i, l = self.randCropSeq(im, label)
            ims_.append(i)
            labels_.append(l)
        return ims_, labels_

    def randCropSeq(self, image, label): #, scale, hval=None, wval=None):
        '''Crop a photo based on input parameters. Allows one to choose a random crop
            and apply it to an entire sequence (for RNN)'''
        hval, wval = self.hval, self.wval
        scale = self.scale
        dst_h, dst_w = self.size, self.size
        if self.aug_type == 1:
            resize_w, resize_h = int(scale*dst_w*1.6), int(scale*dst_h*1.2)
            image = image.resize((resize_w, resize_h), Image.ANTIALIAS)
            label = label.resize((resize_w, resize_h), Image.NEAREST)

            im_w, im_h = image.size[0], image.size[1]
            start_h = int((im_h - dst_h)*hval)
            start_w = int((im_w - dst_w)*wval)
            image = image.crop((start_w, start_h, start_w+dst_w, start_h+dst_h))
            label = label.crop((start_w, start_h, start_w+dst_w, start_h+dst_h))
        else:
            image = image.resize((dst_w, dst_h), Image.ANTIALIAS)
            label = label.resize((dst_w, dst_h), Image.NEAREST)
        image = np.array(image, dtype=np.float32)
        image -= 128 #Assume 128 is the mean.... Weak

#        label = label.resize((self.lbsize, self.lbsize), Image.NEAREST)
        label = np.array(label, dtype=np.int32)
        label[label == 255] = self.ignore_label 
        image = np.expand_dims(image, axis=0)
#        image = torch.from_numpy(image.transpose(2,0,1)).float()
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        return image.div(255), label



    def getparam(self, label):
        '''Used to find height @ which the road stops (vertically seeking)'''
        h, w = label.size()
       # lb = np.zeros([1,w])
        lb = np.zeros([2,w])
#        print "label:\n{}".format(label)
        for i in range(w):
            edge=False
            for j in range(h):
                if not edge: 
                    if label[j, i] == 1:
                        #lb[0, i] = j
                        lb[0, i] = j
                        edge = True
                        #break
                else:
                    if label[j,i] != 1: # The top of the road
                        lb[0, i] = j
                        break
#        print lb
        return torch.from_numpy(lb).float() 

    def lb_to_dist(self, lb):
        '''gt used for experimenting with greater localization (i.e. loss weighted for distance from gt road blob)'''
        h, w = lb.size()
        newlb = np.ones([h, w]) * 255
        for i in range(w):
            dists = []
            for j in range(h-1, 0, -1):
                if lb[j, i] == 0 and lb[j-1, i] == 1:
                    dists.append(j-1)
                elif lb[j, i] == 1 and lb[j-1, i] == 0:
                    dists.append(j)
            idx = h-1
            for l,d in enumerate(dists):
                if l == len(dists)-1:
                    for k in range(idx, -1, -1):
                        newlb[k, i] = d-k
                else:
                    d2 = dists[l+1]
                    for k in range(idx, -1, -1):
                        if abs(d-k) < abs(d2-k):
                            newlb[k, i] = d-k
                        else:
                            idx = k
                            break
        return torch.from_numpy(newlb).float()

