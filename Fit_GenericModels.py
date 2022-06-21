#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import getData
import GenericModels as GM
import FittingLibrary as FL


def Neg2LogLikelihood(theta, data, exp):
    if ProfileLikelihood:
        theta = np.concatenate((theta[:profileIndex], [profileValue], theta[profileIndex:]))
    sigma2, theta = theta[-1], theta[:-1]
    return np.sqrt( np.square(Objective(theta, data, exp))/sigma2 + np.log(sigma2) )


def Objective(theta, data, exp):
    theta = np.exp(theta)
    dose = data[0]
    yData = data[1]
    yModel = GM.FunctionType(theta, dose, exp)
    diff = yModel - yData
    return diff[~np.isnan(diff)]

#######################################################################################################################

def Fitting(runs, kinase, exp, max_nfev=10000, verbose=0):
    dataAll = getData.ReactionVelocity_DDX(kinase)
    path = 'FitResults/Phenomenological_DDX_' + kinase + '_' + exp + '.txt'
    fit_kwargs = {'max_nfev': max_nfev, 'verbose': verbose}
    for ddx in range(5):
        f = open(path, 'a')
        print(ddx)
        data = (dataAll[exp][exp], dataAll[exp]['values'][:,ddx])
        lb, ub = setBounds(exp)
        results = FL.Fitting(Objective, data, exp, runs, lb=lb, ub=ub, fit_kwargs=fit_kwargs)
        results = np.concatenate((results, np.full((results.shape[0],1), ddx)), axis=1)
        np.savetxt(f, results)
        f.close()

#######################################################################################################################

def setBounds(exp):
    if exp in ['ATP', 'pepSub']:
        lb = np.array([1e1, 1, 1e2])
        ub = np.array([1e3, 1e3, 1e5])
    if exp in ['ADP', 'pepMut']:
        lb = np.array([100, 100])
        ub = np.array([300, 1000])
    return np.log(lb), np.log(ub)

#######################################################################################################################

def ProfileLikelihoods(kinase, exp):
    dataAll = getData.ReactionVelocity_DDX(kinase)
    theta, Res2 = PhenomenologicalDDX(kinase, exp)
    fit_kwargs = {'max_nfev': max_nfev, 'verbose': verbose}
    PATH = 'FitResults/Phenomenological_DDX_'
    for ddx in range(5):
        data = (dataAll[exp][exp], dataAll[exp]['values'][:,ddx])
        NN = len(data[0])
        sigma2 = Res2[ddx] / NN
        MLE = np.append(theta, sigma2)
        LLmin = NN * (1 + np.log(sigma2))
        f_profiles = open(PATH + 'Profiles_' + kinase + '_' + ddx + '.txt', 'a')
        f_theta  = open(PATH + 'PLparameters_' + kinase + '_' + ddx + '.txt', 'a')
        for profileIndex in range(len(MLE)):
            print('Processing parameter {} / {}'.format(profileIndex+1, len(MLE)))
            profile, PLparameter = FL.ProfileParameter(Neg2LogLikelihood, exp, data, MLE, LLmin, profileIndex, fit_kwargs, 'detail')
            np.savetxt(f_profiles, profile)
            np.savetxt(f_theta, PLparameter)
        f_profiles.close()
        f_theta.close()
