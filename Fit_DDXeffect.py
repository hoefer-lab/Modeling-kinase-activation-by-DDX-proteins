#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import least_squares
import FittingLibrary as FL
import KinaseModel as KM
import DDXimpact as DDX
import getData
import getFitResults
import importlib


def Objective(thetaDDX, data, modelSpec, ProfileLikelihood=False, profileIndex=None, profileValue=None):
    if ProfileLikelihood:
        thetaDDX = np.concatenate((thetaDDX[:profileIndex], [profileValue], thetaDDX[profileIndex:]))
    thetaDDX = np.exp(thetaDDX)
    if modelSpec.index != -1:
        thetaDDX[1+modelSpec.index] *= thetaDDX[5+modelSpec.index]

    sigma_min = 1e-4
    sigma = thetaDDX[0]
    thetaDDX = thetaDDX[1:]
    if not modelSpec.modeDDX:
        EffSizeDDX = thetaDDX.reshape(4,-1)
    else:
        EffSizeDDX = np.ones((4,4))
        modeDDX = np.array([int(mode) for mode in modelSpec.modeDDX], dtype=bool)
        if (modeDDX[0] + modeDDX[1] > 0) and (modeDDX[2] + modeDDX[3] == 0):
            EffSizeDDX[modeDDX] = np.concatenate((thetaDDX, thetaDDX, np.ones(4), np.ones(4))).reshape(4,4)[modeDDX]
        if (modeDDX[0] + modeDDX[1] == 0) and (modeDDX[2] + modeDDX[3] > 0):
            EffSizeDDX[modeDDX] = np.concatenate((np.ones(4), np.ones(4), thetaDDX, thetaDDX)).reshape(4,4)[modeDDX]
        if (modeDDX[0] + modeDDX[1] > 0) and (modeDDX[2] + modeDDX[3] > 0):
            EffSizeDDX[modeDDX] = np.concatenate((thetaDDX[:4], thetaDDX[:4], thetaDDX[4:], thetaDDX[4:])).reshape(4,4)[modeDDX]

    obj = []
    for exp, dd in data.items():
        E = dd['kinase']
        if exp == 'ATP':
            AA = dd['ATP'][:, np.newaxis]
            BB = dd['pepSub'][0]
        if exp == 'pepSub':
            BB = dd['pepSub'][:, np.newaxis]
            AA = dd['ATP'][0]
        yData = dd['values']
        theta = DDX.setDDXimpact(modelSpec.theta, EffSizeDDX)
        v0 = KM.ReactionVelocity(E, AA, BB, 0, 0, theta)
        yModel = v0[:,1:]/v0[:,0][:,np.newaxis]
        llhood = np.sqrt(2) * np.sqrt( ((yData - yModel) / (sigma + sigma_min))**2 + 2*np.log(sigma/sigma_min + 1) )
        llhood = llhood[ ~np.isnan(llhood) ]
        obj.append(llhood.flatten())

    return np.concatenate(obj)

#######################################################################################################################

def Fitting(runs, kinase, modeDDX=None, procedure='fit', max_nfev=10000, verbose=0):
    path = 'FitResults/DDXeffect_' + kinase
    if modeDDX: path += '_' + modeDDX
    path += '.txt'
    data = getData.ReactionVelocity_DDX(kinase, fold=True)
    data = {exp: data[exp] for exp in ['ATP', 'pepSub']}

    theta = getFitResults.ReactionVelocity(kinase)[0]
    theta = KM.setParameters(theta)
    modelSpec = modelSetup(theta, modeDDX)

    fit_kwargs = {'max_nfev': max_nfev, 'verbose': verbose}
    if procedure == 'fit':
        lb, ub = setBounds(modeDDX)
        FL.Fitting(Objective, data, modelSpec, runs, lb=lb, ub=ub, path=path, fit_kwargs=fit_kwargs)
    elif procedure == 'refit':
        for j in range(runs):
            MLE, LLmin = getFitResults.DDXeffect(kinase, modeDDX)[-2:]
            theta0 = np.random.normal(MLE, scale=0.2)
            FL.Fitting(Objective, data, modelSpec, theta0=theta0, path=path, fit_kwargs=fit_kwargs)

#######################################################################################################################

class modelSetup:
    def __init__(self, theta, modeDDX=None, index=-1):
        self.theta = theta
        self.modeDDX = modeDDX
        self.index = index

#######################################################################################################################

def setBounds(modeDDX):
    if modeDDX:
        if modeDDX=='0000':
            lb = np.log( np.array([0.1]) )
            ub = np.log( np.array([1]) )
        elif (modeDDX[:2]=='00') or (modeDDX[2:]=='00'):
            lb = np.log( np.array([0.1] + 4*[1e-1]) )
            ub = np.log( np.array([1] + 4*[2e1]) )
        else:
            lb = np.log( np.array([0.1] + 8*[1e-1]) )
            ub = np.log( np.array([1] + 8*[2e1]) )
    else:
        lb = np.log( np.array([0.1] + 16*[1e-1]) )
        ub = np.log( np.array([1] + 16*[1e1]) )
    return lb, ub

#######################################################################################################################

def ProfileLikelihoods(kinase, modeDDX=None, max_nfev=10000, verbose=0):
    data = getData.ReactionVelocity_DDX(kinase, fold=True)
    data = {exp: data[exp] for exp in ['ATP', 'pepSub']}

    theta = getFitResults.ReactionVelocity(kinase)[0]
    thetaBind = KM.setParameters(theta)
    MLE, LLmin = getFitResults.DDXeffect(kinase, modeDDX)[-2:]
    modelSpec = modelSetup(thetaBind, modeDDX)

    fit_kwargs = {'max_nfev': max_nfev, 'verbose': verbose}
    filename = kinase
    if modeDDX: filename += '_' + modeDDX
    PATH = 'FitResults/DDXeffect_'
    f_profiles = open(PATH + 'Profiles_' + filename + '.txt', 'a')
    f_theta  = open(PATH + 'PLparameters_' + filename + '.txt', 'a')
    for profileIndex in range(len(MLE)):
        print('Processing parameter {} / {}'.format(profileIndex+1, len(MLE)))
        profile, PLparameter = FL.ProfileParameter(Objective, modelSpec, data, MLE, LLmin, profileIndex, fit_kwargs, 'detail')
        np.savetxt(f_profiles, profile)
        np.savetxt(f_theta, PLparameter)
    if modeDDX and len(MLE) > 5:
        # profile Likelihoods for effect size ratios
        for index in range(4):
            modelSpec.index = index
            MLEmod = np.copy(MLE)
            MLEmod[1+index] -= MLEmod[5+index]
            print('Processing parameter {}'.format(index+1))
            profile, PLparameter = FL.ProfileParameter(Objective, modelSpec, data, MLEmod, LLmin, 1+index, fit_kwargs, disp='detail')
            np.savetxt(f_profiles, profile)
            np.savetxt(f_theta, PLparameter)
    f_profiles.close()
    f_theta.close()

#######################################################################################################################
