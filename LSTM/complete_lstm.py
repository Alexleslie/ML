import numpy as np
import random

def sigmoid(data):
    return 1. / (1 + np.exp(-data))
   
def rand_arr(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a
    
