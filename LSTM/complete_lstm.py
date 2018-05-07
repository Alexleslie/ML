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


class LstmState:
    def __init__(self, mem_cell_bt, x_dim):
        self.g = np.zeros(mem_cell_bt)
        self.i = np.zeros(mem_cell_bt)
        self.f = np.zeros(mem_cell_bt)
        self.o = np.zeros(mem_cell_bt)
        self.s = np.zeros(mem_cell_bt)
        self.h = np.zeros(mem_cell_bt)

        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)
        self.bottom_diff_x = np.zeros(x_dim)


class LstmNode:
    def __init__(self, lstm_param, lstm_state):
        self.lstm_param = lstm_param
        self.lstm_state = lstm_state

        self.x = None
        self.xc = None

    def bottom_data_is(self, x, s_prev=None, h_prev=None):
        if s_prev == None: self.s_prev = np.zeros_like(self.lstm_state.s)
        if h_prev == None: self.h_prev = np.zeros_like(self.lstm_state.h)

        self.s_prev = s_prev
        self.h_prev = h_prev

        xc = np.hstack((x, h_prev))

        self.lstm_state.g = np.tanh(np.dot(self.lstm_param.wg, xc) + self.lstm_param.bg)
        self.lstm_state.i = sigmoid(np.dot(self.lstm_param.wi, xc) + self.lstm_param.bi)
        self.lstm_state.f = sigmoid(np.dot(self.lstm_param.wf, xc) + self.lstm_param.bf)
        self.lstm_state.o = sigmoid(np.dot(self.lstm_param.wo, xc) + self.lstm_param.bo)

        self.lstm_state.s = self.lstm_state.g * self.lstm_state.i + s_prev * self.lstm_state.f
        self.lstm_state.h = self.lstm_state.s * self.lstm_state.o

        self.x = x
        self.xc = xc

    def top_diff_is(self, top_diff_h, top_diff_s):
        ds = self.lstm_state.o * top_diff_h + top_diff_s
        do = self.lstm_state.s * top_diff_h
        di = self.lstm_state.g * ds
        dg = self.lstm_state.i * ds
        df = self.s_prev * ds

        di_input = (1. - self.lstm_state.i) * self.lstm_state.i * di
        df_input = (1. - self.lstm_state.f) * self.lstm_state.f * df
        do_input = (1. - self.lstm_state.o) * self.lstm_state.o * do
        dg_input = (1. - self.lstm_state.g**2) * dg

        self.lstm_param.wi_diff += np.outer(di_input, self.xc)
        self.lstm_param.wf_diff += np.outer(df_input, self.xc)
        self.lstm_param.wo_diff += np.outer(do_input, self.xc)
        self.lstm_param.wg_diff += np.outer(dg_input, self.xc)

        self.lstm_param.bi_diff += di_input
        self.lstm_param.bf_diff += df_input
        self.lstm_param.bo_diff += do_input
        self.lstm_param.bg_diff += dg_input

        dxc = np.zeros(self.xc)
        dxc += np.dot(self.lstm_param.wi.T, di_input)
        dxc += np.dot(self.lstm_param.wf.T, df_input)
        dxc += np.dot(self.lstm_param.wo.T, do_input)
        dxc += np.dot(self.lstm_param.wg.T, dg_input)

        self.lstm_param.bottom_diff_s = ds * self.lstm_state.f
        self.lstm_param.bottom_diff_h = dxc[self.lstm_param.x_dim:]


class LstmNetwork:
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        self.x_list = []

    def y_list_is(self, y_list, loss_layer):

        assert len(y_list) == len(loss_layer)
        idx = len(self.x_list) - 1
        loss = loss_layer.loss(self.lstm_node_list[idx], y_list[idx])

        diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
        diff_s = np.zeros(self.lstm_param.mem_cell_ct)
        self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
        idx -= 1
        while idx >= 0:
            loss += loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s

            self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)

            idx -= 1
        return loss

    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))

        idx = len(self.x_list) - 1
        if idx == 0:
            self.lstm_node_list[idx].bottom_data_is(x)
        else:
            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)








