# -*- coding: utf-8 -*-
"""
@Time ： 2020/07/25 15:39
@Auth ： Kunfeng Li
@File ：vdn_mixer.py
@IDE ：PyCharm

"""
import torch.nn as nn
import torch


class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, q_values, s):
        return torch.sum(q_values, dim=2, keepdim=True)
