import torch.utils.data as data
import torch
import numpy as np
import numpy.random as rand
from PIL import Image
import cv2
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(txtdir, root):
    images = []

    with open(txtdir) as f:
        for line in f:
            parts = line.rstrip().split('||')
            if is_image_file(parts[0]):
                images.append((os.path.join(root, parts[0]), parts[1]))

    return images


class NetLoader(data.Dataset):

    def __init__(self, txtdir, root, aug_type=0, transform=None, target_transform=None, ignore_label=255, tosize=224, lbsize=None, lb_type='map'):
        imgs = make_dataset(txtdir, root)
        
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.aug_type = aug_type
        self.transform = transform
        self.target_transform = target_transform
        self.size = tosize
        if lbsize is None:
            self.lbsize = tosize
        else:
            self.lbsize = lbsize
        self.ignore_label = ignore_label
        self.lb_type = lb_type

    def __getitem__(self, index):
        impath, lbpath = self.imgs[index]
        img, lb = Image.open(impath).convert('RGB'), Image.open(lbpath)

        if self.aug_type > 0:
            img, lb = self.randomcrop(img, lb)
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
        return len(self.imgs)

    def randomcrop(self, image, label):
        dst_h, dst_w = self.size, self.size

        if self.aug_type == 1:
            #random scale
            scale = 0.9 + rand.random() * 0.2
            resize_w, resize_h = int(scale*dst_w*1.6), int(scale*dst_h*1.2)
            image = image.resize((resize_w, resize_h), Image.ANTIALIAS) # Image.LANCZOS)
            label = label.resize((resize_w, resize_h), Image.NEAREST)

            #random crop
            im_w, im_h = image.size[0], image.size[1]
            start_h = int((im_h - dst_h)*rand.random())
            start_w = int((im_w - dst_w)*rand.random())
            image = image.crop((start_w, start_h, start_w+dst_w, start_h+dst_h))
            label = label.crop((start_w, start_h, start_w+dst_w, start_h+dst_h))
        else:
            image = image.resize((self.size, self.size), Image.ANTIALIAS) #Image.LANCZOS) ERR:Module not found
            label = label.resize((self.size, self.size), Image.NEAREST)

        image = np.array(image, dtype=np.float32)
        image -= 128 #Assume 128 is the mean.... Weak

        label = label.resize((self.lbsize, self.lbsize), Image.NEAREST)
        label = np.array(label, dtype=np.int32)
        label[label == 255] = self.ignore_label 
        image = torch.from_numpy(image.transpose(2,0,1)).float()
        label = torch.from_numpy(label).long()
        return image.float().div(255), label

    def getparam(self, label):
        h, w = label.size()
        lb = np.zeros([1,w])
        for i in range(w):
            for j in range(h-1, -1, -1):
                if label[j, i] == 0:
                    lb[0, i] = j
                    break
        return torch.from_numpy(lb).float() 

    def lb_to_dist(self, lb):
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

