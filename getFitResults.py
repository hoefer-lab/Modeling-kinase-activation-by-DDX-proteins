#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import KinaseModel as KM
import DDXimpact

#######################################################################################################################
# Fits DataMatrices
#######################################################################################################################

def ReactionVelocity(kinase):
    path = 'FitResults/ReactionVelocity_' + kinase + '.txt'
    results = pd.read_csv(path, header=None, sep=' ').values
    # log(MLE) = results[:, :-2], log(LLmin) = results[:, -2], exitflag = results[:, -1]
    results = results[results[:,-2] > 0]
    LLmin = np.min(results[:,-2])
    MLE = results[ np.argmin(results[:,-2]), :-2 ]
    CV = np.exp(MLE[0])
    theta = np.exp(MLE[1:])
    return theta, CV, MLE, LLmin

#######################################################################################################################

def ReactionVelocity_Profile(index, kinase, cutoff=None):
    path = 'FitResults/ReactionVelocity_Profiles_' + kinase + '.txt'
    profile = pd.read_csv(path, header=None, sep=' ', skiprows=index, nrows=2)
    theta = np.exp(profile.values[0])
    LL = profile.values[1]
    if cutoff:
        theta, LL = theta[LL < cutoff], LL[LL < cutoff]
    CI = LL < 3.841
    return theta, LL, CI

#######################################################################################################################
# Fits DDX
#######################################################################################################################

def DDXeffect(kinase, modeDDX=None):
    path = 'FitResults/DDXeffect_' + kinase
    if modeDDX:
        path += '_' + modeDDX +  '.txt'
    else:
        path += '.txt'
    results = pd.read_csv(path, header=None, sep=' ').values
    # log(MLE) = results[:, :-2], log(LLmin) = results[:, -2], exitflag = results[:, -1]
    results = results[ results[:,-2] > 0 ]
    LLmin = np.min(results[:,-2])
    MLE = results[ np.argmin(results[:,-2]), :-2 ]
    sigma = np.exp(MLE[0])
    facDDX = np.exp(MLE[1:])
    return facDDX, sigma, MLE, LLmin

#######################################################################################################################

def DDXeffect_Akaike(kinase):
    AICc = {}
    AICc_best = np.inf
    for ATP in ['00', '10', '01', '11']:
        for pepSub in ['00', '10', '01', '11']:
            modeDDX = ATP + pepSub
            facDDX, sigma, MLE, LLmin = DDXeffect(kinase, modeDDX)
            AICc.update({modeDDX: (LLmin + 2*len(MLE), facDDX)})
            if AICc[modeDDX][0] < AICc_best:
                AICc_best = AICc[modeDDX][0]
    return AICc

#######################################################################################################################

def DDXeffect_Profile(index, kinase, modeDDX=None, cutoff=None):
    path = 'FitResults/DDXeffect_Profiles_' + kinase
    if modeDDX:
        path += '_' + modeDDX +  '.txt'
    else:
        path += '.txt'
    profile = pd.read_csv(path, header=None, sep=' ', skiprows=index, nrows=2)
    theta = np.exp(profile.values[0])
    LL = profile.values[1]
    if cutoff:
        theta, LL = theta[LL < cutoff], LL[LL < cutoff]
    CI = LL < 3.841
    return theta, LL, CI

#######################################################################################################################

def DDXeffect_CIs(kinase):
    CI = {}
    indices = [('nucleotides', 'affinity'), ('nucleotides', 'off-rate'), ('peptides', 'affinity'), ('peptides', 'off-rate'), ('nucleotides', 'on-rate'), ('peptides', 'on-rate')]
    for ind, index in enumerate(indices):
        CIDDX = np.zeros((4,2))
        for d in range(4):
            theta, LL, CIind = DDXeffect_Profile(2*(4*ind+d+1), kinase)
            CIDDX[d] = theta[CIind][[0,-1]]
        CI[index] = CIDDX
    return CI

#######################################################################################################################

def PhenomenologicalDDX(kinase, exp):
    path = 'FitResults/Phenomenological_DDX_' + kinase + '_' + exp + '.txt'
    if exp in ['ATP', 'pepSub']:
        theta = np.zeros((3, 5))
    if exp in ['ADP', 'pepMut']:
        theta = np.zeros((2, 5))
    LLmin = np.zeros(5)
    results = pd.read_csv(path, header=None, sep=' ').values
    for ddx in range(5):
        res = results[results[:,-1]==ddx, :-1]
        res = res[ res[:,-2] > 0 ]
        LLmin[ddx] = np.min(results[:,-2])
        theta[:, ddx] = np.exp(res[ np.argmin(res[:,-2]), :-2 ])
    return theta, LLmin

#######################################################################################################################

def DDXdose(kinase, modeDDX):
    path = 'FitResults/DDXdose_' + kinase + '_' + modeDDX + '.txt'
    results = pd.read_csv(path, header=None, sep=' ').values
    results = results[ results[:,-2] > 0 ]
    LLmin = np.min(results[:,-2])
    MLE = results[ np.argmin(results[:,-2]), :-2 ]
    sigma = np.exp(MLE[0])
    delRate = np.exp(MLE[1])
    KDDX = np.exp(MLE[2:])
    return KDDX, delRate, sigma, MLE, LLmin

#######################################################################################################################

def DDXdose_Profile(index, kinase, cutoff=None):
    modeDDX = DDXdose(kinase)[0]
    path = 'FitResults/DDXdose_Profiles_'  + kinase + '_' + modeDDX + '.txt'
    profile = pd.read_csv(path, header=None, sep=' ', skiprows=index, nrows=2)
    theta = np.exp(profile.values[0])
    LL = profile.values[1]
    if cutoff:
        theta, LL = theta[LL < cutoff], LL[LL < cutoff]
    CI = LL < 3.841
    return theta, LL, CI

#######################################################################################################################
