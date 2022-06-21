#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import FittingLibrary as FL
import KinaseModel as KM
import DDXimpact as DDX
import getData
import getFitResults


def Objective(thetaDDX, data, modelSpec, ProfileLikelihood=False, profileIndex=None, profileValue=None):
    if ProfileLikelihood:
        thetaDDX = np.concatenate((thetaDDX[:profileIndex], [profileValue], thetaDDX[profileIndex:]))
    thetaDDX = np.exp(thetaDDX)

    theta0, modeDDX, E, A, B = modelSpec

    sigma_min = 1e-4
    sigma = thetaDDX[0]
    delRate = thetaDDX[1]
    KDDX = np.concatenate( [ thetaDDX[j+2]*data[0] for j in range(4) ] )
    facDDX = np.tile( ( 1 + delRate*KDDX ) / ( 1 + KDDX ), 4).reshape(4,-1)
    EffSizeDDX = np.ones_like(facDDX)
    modeDDX = np.array([int(mode) for mode in modeDDX], dtype=bool)
    EffSizeDDX[modeDDX] = facDDX[modeDDX]

    ydata = data[1].reshape(-1, order='F')
    theta = DDX.setDDXimpact(theta0, EffSizeDDX)
    v0 = KM.ReactionVelocity(E, A, B, 0, 0, theta)
    ymodel = v0[1:] / v0[0]
    obj = np.sqrt(2) * np.sqrt( ((ydata - ymodel) / (sigma + sigma_min))**2 + 2*np.log(sigma/sigma_min + 1) )

    return obj

#######################################################################################################################

def Fitting(runs, kinase, modeDDX, max_nfev=10000, verbose=0):
    path = 'FitResults/DDXdose_' + kinase + '_' + modeDDX + '.txt'
    fit_kwargs = {'max_nfev': max_nfev, 'verbose': verbose}

    data = getData.ReactionVelocity_DDX(kinase, dose=True, fold=True)['Sheet1']

    theta = getFitResults.ReactionVelocity(kinase)[0]
    theta = KM.setParameters(theta)

    modelSpec = (theta, modeDDX, data['kinase'], data['ATP'][0], data['pepSub'][0])

    delRate_lb, delRate_ub = (100, 1000) if kinase == 'CK1' else (2, 5)
    aff_lb, aff_ub = (1e-3, 1e1) if kinase == 'CK1' else (1e-1, 1e2)
    lb = np.log(np.array([0.1] + [delRate_lb] + 4*[aff_lb]))
    ub = np.log(np.array([0.5] + [delRate_ub] + 4*[aff_ub]))
    FL.Fitting(Objective, (data['DDX'], data['values']), modelSpec, runs, lb=lb, ub=ub, path=path, fit_kwargs=fit_kwargs)


#######################################################################################################################

def ProfileLikelihoods(kinase, modeDDX):
    data = getData.ReactionVelocity_DDX(kinase, dose=True, fold=True)
    data = data['Sheet1'] if kinase == 'CK1' else data['Sheet2']
    theta = getFitResults.ReactionVelocity(kinase, model)[0]
    thetaBind = KM.setParameters(theta, model)

    KDDX, delRate, sigma, MLE, LLmin = getFitResults.DDXdose(kinase, modeDDX)
    modelSpec = (thetaBind, modeDDX, data['kinase'], data['ATP'][0], data['pepSub'][0])
    data = (data['DDX'], data['values'])

    PATH = 'FitResults/DDXdose_'
    filename = kinase + '_' + modeDDX['ident']

    fit_kwargs = {'verbose': 0, 'max_nfev': 1000}

    FL.ProfileLikelihoods(Objective, modelSpec, data, MLE, LLmin, PATH, filename, fit_kwargs, disp='detail')

#######################################################################################################################
