import numpy as np
import random
import math


def sigmoid(data):
    return 1. / (1 + np.exp(-data))
   

def rand_arr(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a


class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = mem_cell_ct + x_dim

        self.wg = rand_arr(-1.0, 1.0, (self.mem_cell_ct, concat_len))
        self.wi = rand_arr(-1.0, 1.0, (self.mem_cell_ct, concat_len))
        self.wf = rand_arr(-1.0, 1.0, (self.mem_cell_ct, concat_len))
        self.wo = rand_arr(-1.0, 1.0, (self.mem_cell_ct, concat_len))

        self.bg = rand_arr(-1.0, 1.0, mem_cell_ct)
        self.bi = rand_arr(-1.0, 1.0, mem_cell_ct)
        self.bf = rand_arr(-1.0, 1.0, mem_cell_ct)
        self.bo = rand_arr(-1.0, 1.0, mem_cell_ct)

        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)

        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)

    def apply_diff(self, lr=1):
        self.wg += lr * self.wg_diff
        self.wi += lr * self.wi_diff
        self.wf += lr * self.wf_diff
        self.wo += lr * self.wo_diff

        self.bg += lr * self.bg_diff
        self.bi += lr * self.bi_diff
        self.bf += lr * self.bf_diff
        self.bo += lr * self.bo_diff

        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)

        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)



