#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

########################################################################################################################

def setDDXimpact(theta0, EffSizeDDX):
    theta = np.repeat(theta0[:,:,np.newaxis], EffSizeDDX.shape[1]+1, axis=2)
    vartheta = np.ones((3,5, EffSizeDDX.shape[1]))
    paraType = getParaTypes()
    # DDX effect size:
    # [0] -> on-rate nucleotides
    # [1] -> off-rate nucleotides
    # [2] -> on-rate peptides
    # [3] -> off-rate peptides
    # paraTypes:
    # affinity = 1, off-rate=2, on-rate=3
    # nucleotides (+), peptides (-)
    vartheta[paraType==1] = EffSizeDDX[0] / EffSizeDDX[1]
    vartheta[paraType==2] = EffSizeDDX[1]
    vartheta[paraType==3] = EffSizeDDX[0]
    vartheta[paraType==-1] = EffSizeDDX[2] / EffSizeDDX[3]
    vartheta[paraType==-2] = EffSizeDDX[3]
    vartheta[paraType==-3] = EffSizeDDX[2]
    theta[:,:,1:] *= vartheta

    return theta

########################################################################################################################

def getParaTypes():
    #   / K_1  k_-1  k_3  k_6  --- \
    #  |  K_2  k_-2  k_4  k_5  K_7  |
    #   \ K_6  K_8   K_9  K_10 --- /

    paraType = np.zeros((3,5), dtype=int)
    # affinity = 1, off-rate=2, on-rate=3
    # nucleotides (+), peptides (-)
    paraType[[0,2], 0] = 1
    paraType[1, [0,4]] = -1
    paraType[2, 1:4] = -1
    paraType[0, [1,3]] = 2
    paraType[1, [1,3]] = -2
    paraType[0, 2] = 3
    paraType[1, 2] = -3
    return paraType
