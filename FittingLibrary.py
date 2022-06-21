#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import least_squares

#######################################################################################################################
# Fitting
#######################################################################################################################

def MinimizeObjective(ObjFunc, theta0, data, ModelFunc, fit_kwargs={}, **kwargs):
    fit = least_squares(ObjFunc, theta0, args=(data, ModelFunc), kwargs=kwargs, **fit_kwargs)
    return fit

#######################################################################################################################

def Fitting(ObjFunc, data, ModelFunc, runs=1, lb=None, ub=None, theta0=None, path=None, fit_kwargs={}, **kwargs):
    theta0 = theta0.reshape(1,-1) if theta0 is not None else LatinHypercubeSampling(runs, lb, ub)
    results = np.zeros((runs, theta0.shape[1] + 2))
    for j in range(runs):
        fit = MinimizeObjective(ObjFunc, theta0[j], data, ModelFunc, fit_kwargs, **kwargs)
        results[j] = np.concatenate((fit.x, [fit.cost, fit.status]))
        print('Termination for Run {}/{}:'.format(j+1, runs), fit.message)
        print('Final cost: ', fit.cost)
        if path:
            f = open(path, 'a')
            np.savetxt(f, results[j].reshape(1,-1))
            f.close()
    return results

#######################################################################################################################
# ParaSpace sampling
#######################################################################################################################

def LatinHypercubeSampling(runs, lb, ub):

    def ParaSpaceDecomposition(runs, lb, ub):
        thetaGrid = np.zeros((numParas, runs+1))
        for j in range(numParas):
            thetaGrid[j] = np.linspace(lb[j], ub[j], runs+1)
        return thetaGrid

    numParas = len(lb)
    thetaGrid = ParaSpaceDecomposition(runs, lb, ub)
    interval = np.zeros((numParas, runs), dtype=int)
    for k in range(numParas):
        interval[k] = np.random.permutation(runs)

    ic = np.zeros((runs, numParas))
    for k in range(runs):
        for m in range(numParas):
            a = thetaGrid[m, interval[m, k]]
            b = thetaGrid[m, interval[m, k]+1]
            ic[k, m] = a + (b-a)*np.random.rand()
    return ic

#######################################################################################################################
# Profile Likelihoods
#######################################################################################################################

def ProfileLikelihoods(ObjFunc, ModelFunc, data, MLE, LLmin, PATH, filename, fit_kwargs={}, disp='step', **kwargs):
    f_profiles = open(PATH + 'Profiles_' + filename + '.txt', 'a')
    f_theta  = open(PATH + 'PLparameters_' + filename + '.txt', 'a')

    for profileIndex in range(len(MLE)):
        if (disp == 'step') or (disp == 'detail'):
            print('Processing parameter {} / {}'.format(profileIndex+1, len(MLE)))
        profile, PLparameter = ProfileParameter(ObjFunc, ModelFunc, data, MLE, LLmin, profileIndex, fit_kwargs, disp, **kwargs)
        np.savetxt(f_profiles, profile)
        np.savetxt(f_theta, PLparameter)

    f_profiles.close()
    f_theta.close()

#######################################################################################################################

def ProfileParameter(ObjFunc, ModelFunc, data, MLE, LLmin, profileIndex, fit_kwargs={}, disp='detail', **kwargs):
    LLHood_l, profVal_l, theta_l = ProfileParameter_direction(ObjFunc, ModelFunc, data, MLE, LLmin, profileIndex, 'left', fit_kwargs, disp, **kwargs)
    LLHood_r, profVal_r, theta_r = ProfileParameter_direction(ObjFunc, ModelFunc, data, MLE, LLmin, profileIndex, 'right', fit_kwargs, disp, **kwargs)
    LLHood = np.concatenate((LLHood_l[::-1], [0], LLHood_r))
    profVal = np.concatenate((profVal_l[::-1], [MLE[profileIndex]], profVal_r))
    profile = np.vstack((profVal, LLHood))
    PLparameter = np.concatenate((theta_l[:, ::-1], MLE[:,np.newaxis], theta_r), axis=1)

    return profile, PLparameter

#######################################################################################################################

def ProfileParameter_direction(ObjFunc, ModelFunc, data, MLE, LLmin, profileIndex, direction, fit_kwargs={}, disp='step', maxIter=500, DeltaLLHood_max=5, profileVal_max=20, stepStart=0.001, stepIncr=1.5, stepDecr=0.7, LLHoodIncr=0.01, LLHoodDecr=0.05):
    sign = 1 if direction == 'right' else -1

    profileVal = MLE[profileIndex]
    if sign*profileVal < profileVal_max:
        theta0 = MLE[np.arange(len(MLE)) != profileIndex]
        LLHood_old = LLmin
        profileVal_old = profileVal
        LLHood_out = np.zeros(maxIter)
        profileVal_out = np.zeros(maxIter)
        theta_out = np.zeros((len(MLE), maxIter))
        progress = True
        iteration = 0
        step = stepStart
    else:
        LLHood_out = np.empty(0)
        profileVal_out = np.empty(0)
        theta_out = np.empty((len(MLE), 0))
        progress = False

    while progress:
        repeat = True

        profileVal = profileVal_old + sign*step
        while repeat:
            fit = MinimizeObjective(ObjFunc, theta0, data, ModelFunc, fit_kwargs, ProfileLikelihood=True, profileIndex=profileIndex, profileValue=profileVal)
            theta = fit.x
            LLHood = fit.cost
            if np.abs(LLHood - LLHood_old) < LLHoodIncr:
                step *= stepIncr
                repeat = False
            elif (LLHood - LLHood_old) > LLHoodDecr:
                step *= stepDecr
                profileVal = profileVal_old + sign*step
            else:
                repeat = False

        theta0 = theta
        LLHood_old = LLHood
        profileVal_old = profileVal

        if disp == 'detail': print(LLHood - LLmin)

        LLHood_out[iteration] = LLHood - LLmin
        profileVal_out[iteration] = profileVal
        theta_out[:, iteration] = np.concatenate((theta[:profileIndex], [profileVal], theta[profileIndex:]))

        iteration += 1

        if (iteration == maxIter) or (LLHood - LLmin > DeltaLLHood_max) or (sign*profileVal > profileVal_max):
            progress = False
            LLHood_out = LLHood_out[:iteration]
            profileVal_out = profileVal_out[:iteration]
            theta_out = theta_out[:, :iteration]

    return LLHood_out, profileVal_out, theta_out

#######################################################################################################################
# Prediction Profile Likelihoods
#######################################################################################################################

def PredictionProfileLikelihood(ObjFunc, ModelFunc, data, MLE, LLmin, PATH, filename, fit_kwargs={}, disp='step', **kwargs):
    f_profiles = open(PATH + 'Prediction_' + filename + '.txt', 'a')
    f_theta  = open(PATH + 'PredictionParameters_' + filename + '.txt', 'a')

    LLHood_l, profVal_l, theta_l = PredictionProfileParameter_direction(ObjFunc, ModelFunc, data, MLE, LLmin, profileIndex, 'left', fit_kwargs, disp, **kwargs)
    LLHood_r, profVal_r, theta_r = PredictionProfileParameter_direction(ObjFunc, ModelFunc, data, MLE, LLmin, profileIndex, 'right', fit_kwargs, disp, **kwargs)
    LLHood = np.concatenate((LLHood_l[::-1], [0], LLHood_r))
    profVal = np.concatenate((profVal_l[::-1], [MLE[profileIndex]], profVal_r))
    profile = np.vstack((profVal, LLHood))
    PLparameter = np.concatenate((theta_l[:, ::-1], MLE[:,np.newaxis], theta_r), axis=1)

    np.savetxt(f_profiles, profile)
    np.savetxt(f_theta, PLparameter)

    f_profiles.close()
    f_theta.close()

#######################################################################################################################

def PredictionProfileParameter_direction(ObjFunc, ModelFunc, data, MLE, LLmin, profileIndex, direction, fit_kwargs={}, disp='step', maxIter=500, DeltaLLHood_max=5, profileVal_max=20, stepStart=0.01, stepIncr=1.5, stepDecr=0.7, LLHoodIncr=0.01, LLHoodDecr=0.05):
    sign = 1 if direction == 'right' else -1

    profileVal = MLE[profileIndex]
    if sign*profileVal < profileVal_max:
        theta0 = MLE[np.arange(len(MLE)) != profileIndex]
        LLHood_old = LLmin
        profileVal_old = profileVal
        LLHood_out = np.zeros(maxIter)
        profileVal_out = np.zeros(maxIter)
        theta_out = np.zeros((len(MLE), maxIter))
        progress = True
        iteration = 0
        step = stepStart
    else:
        LLHood_out = np.empty(0)
        profileVal_out = np.empty(0)
        theta_out = np.empty((len(MLE), 0))
        progress = False

    while progress:
        repeat = True

        profileVal = profileVal_old + sign*step
        while repeat:
            fit = MinimizeObjective(ObjFunc, theta0, data, ModelFunc, fit_kwargs, ProfileLikelihood=True, profileIndex=profileIndex, profileValue=profileVal)
            theta = fit.x
            LLHood = fit.cost
            if np.abs(LLHood - LLHood_old) < LLHoodIncr:
                step *= stepIncr
                repeat = False
            elif (LLHood - LLHood_old) > LLHoodDecr:
                step *= stepDecr
                profileVal = profileVal_old + sign*step
            else:
                repeat = False

        theta0 = theta
        LLHood_old = LLHood
        profileVal_old = profileVal

        if disp == 'detail': print(LLHood - LLmin)

        LLHood_out[iteration] = LLHood - LLmin
        profileVal_out[iteration] = profileVal
        theta_out[:, iteration] = np.concatenate((theta[:profileIndex], [profileVal], theta[profileIndex:]))

        iteration += 1

        if (iteration == maxIter) or (LLHood - LLmin > DeltaLLHood_max) or (sign*profileVal > profileVal_max):
            progress = False
            LLHood_out = LLHood_out[:iteration]
            profileVal_out = profileVal_out[:iteration]
            theta_out = theta_out[:, :iteration]

    return LLHood_out, profileVal_out, theta_out

#######################################################################################################################
