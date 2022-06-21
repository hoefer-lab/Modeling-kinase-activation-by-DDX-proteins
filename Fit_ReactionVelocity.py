#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import getData
import getFitResults
import KinaseModel as KM
import FittingLibrary as FL


def Objective(theta, data, modelSpec, ProfileLikelihood=False, profileIndex=None, profileValue=None):
    if ProfileLikelihood:
        theta = np.concatenate((theta[:profileIndex], [profileValue], theta[profileIndex:]))
    theta = np.exp(theta)
    if modelSpec.index != -1: # translate on-rates back to affinities
        if modelSpec.index == 1:
            theta[1] /= theta[2] # K1 = k1/km1
        if modelSpec.index == 3:
            theta[3] /= theta[12] # K2 = k2/km2
        if modelSpec.index == 6:
            theta[6] /= theta[7] # K6 = km6/k6

    sigmaMin = 1e-4
    CV = theta[0]
    theta = KM.setParameters(theta[1:])
    obj = []
    for exp, dd in data.items():
        E = dd['kinase']
        AA = dd['ATP'][np.newaxis, :]
        BB = dd['pepSub'][:, np.newaxis]
        QQ = dd['factor_ADP'] * AA
        MM = dd['factor_pepMut'] * BB
        yData = dd['values']
        yModel = modelSpec.model(E, AA, BB, QQ, MM, theta)
        sigma = CV * yModel
        llhood = np.sqrt(2) * np.sqrt( ((yData - yModel) / (sigma + sigmaMin))**2 + 2*np.log(sigma/sigmaMin + 1) )
        llhood = llhood[ ~np.isnan(llhood) ]
        obj.append(llhood.flatten())
    return np.concatenate(obj)

#######################################################################################################################

def Fitting(runs, kinase, procedure='fit', max_nfev=10000, verbose=0):
    data = getData.ReactionVelocity(kinase)
    modelSpec = modelSetup(KM.ReactionVelocity)
    path = 'FitResults/ReactionVelocity_' + kinase + '.txt'
    fit_kwargs = {'max_nfev': max_nfev, 'verbose': verbose}
    if procedure == 'fit':
        lb, ub = setBounds()
        FL.Fitting(Objective, data, modelSpec, runs, lb=lb, ub=ub, path=path, fit_kwargs=fit_kwargs)
    elif procedure == 'refit':
        for j in range(runs):
            MLE = getFitResults.ReactionVelocity(kinase)[2]
            theta0 = np.random.normal(MLE, scale=0.1)
            FL.Fitting(Objective, data, modelSpec, theta0=theta0, path=path, fit_kwargs=fit_kwargs)

#######################################################################################################################

class modelSetup:
    def __init__(self, model, index=-1):
        self.model = model
        self.index = index

#######################################################################################################################

def setBounds():
    lb = np.array([0.1, 0.1] + 13*[1e-2])
    ub = np.array([0.3, 5] + 13*[1e2])
    return np.log(lb), np.log(ub)

#######################################################################################################################

def ProfileLikelihoods(kinase, max_nfev=2000, verbose=0):
    data = getData.ReactionVelocity(kinase)
    modelSpec = modelSetup(KM.ReactionVelocity)
    MLE, LLmin = getFitResults.ReactionVelocity(kinase)[2:]
    fit_kwargs = {'max_nfev': max_nfev, 'verbose': verbose}
    PATH = 'FitResults/ReactionVelocity_'
    f_profiles = open(PATH + 'Profiles_' + kinase + '.txt', 'a')
    f_theta  = open(PATH + 'PLparameters_' + kinase + '.txt', 'a')
    for profileIndex in range(len(MLE)):
        print('Processing parameter {} / {}'.format(profileIndex+1, len(MLE)))
        profile, PLparameter = FL.ProfileParameter(ObjFunc, model, data, MLE, LLmin, profileIndex, fit_kwargs, 'detail')
        np.savetxt(f_profiles, profile)
        np.savetxt(f_theta, PLparameter)
    # profile Likelihoods for on-rates of ATP, peptide, and ADP binding
    for index in [1, 3, 6]:
        if kinase == 'CK1' and index == 3: continue
        modelSpec.index = index
        MLEmod = np.copy(MLE)
        if index == 1:
            MLEmod[1] += MLEmod[2] # log(k1) = log(K1) + log(km1)
        if index == 3:
            MLEmod[3] += MLEmod[12] # log(k2) = log(K2) + log(km2)
        if index == 6:
            MLEmod[6] += MLEmod[7] # log(km6) = log(K6) + log(k6)
        print('Processing parameter {}'.format(index))
        profile, PLparameter = FL.ProfileParameter(Objective, modelSpec, data, MLEmod, LLmin, index, fit_kwargs, disp='detail')
        np.savetxt(f_profiles, profile)
        np.savetxt(f_theta, PLparameter)

    f_profiles.close()
    f_theta.close()

#######################################################################################################################
