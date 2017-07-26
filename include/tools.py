import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import time

import torch
from torch.autograd import Variable





def maptoparam(target_var):
    road = target_var == 1
    h,w = target_var.data.size()
    params=  torch.zeros(2, w)
    for i in range(w):
        for j in range(h-1, -1, -1):
            if target_var 
