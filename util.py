import numpy as np
import sys
import os
import torch
import pickle

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def onehot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[int(index)] = 1.0
    return vec

