#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#######################################################################################################################

def ReactionVelocity(E, A, S, Q, M, theta):
    #          / K_1  k_-1  k_3  k_6  --- \
    # theta = |  K_2  k_-2  k_4  k_5  K_7  |
    #          \ K_6  K_8   K_9  K_10 --- /
    #
    # ATP binding
    K1 = theta[0,0]
    km1 = theta[0,1]
    k3 = theta[0,2]
    # peptide
    K2 = theta[1,0]
    km2 = theta[1,1]
    k4 = theta[1,2]
    k5 = theta[1,3]
    K7 = theta[1,4]
    # ADP binding
    K6 = theta[2,0]
    k6 = theta[0,3]
    # mutant
    K8 = theta[2,1]
    K9 = theta[2,2]
    K10 = theta[2,3]

    # rescaling
    alp31 = k3 / (K1 * km2)
    alp42 = k4 / (K2 * km1)
    AA = K1*A
    SS = K2*S
    QQ = K6*Q
    MM = K8*M
    XXA = 1 + alp31*AA
    XXS = 1 + alp42*SS
    deadEQS = 1 + K7*S + K10*M

    denom = (k5*deadEQS + k6) * (km2*alp31*XXS + km1*alp42*XXA) * AA * SS + k5*k6 * ( (1 + MM + QQ*deadEQS) * XXS * XXA + AA*(1 + K9*M)*XXA + SS*XXS )
    return E * k5 * k6 * (km2*alp31*XXS + km1*alp42*XXA) * AA * SS / denom

#######################################################################################################################

def setParameters(thetaModel):
    theta = np.zeros((3,5))
    theta[0,0] = thetaModel[0] # K1
    theta[0,1] = thetaModel[1] # km1
    theta[1,0] = thetaModel[2] # K2
    theta[1,2] = thetaModel[3] # k4
    theta[1,4] = thetaModel[4] # K7
    theta[2,0] = thetaModel[5] # K6
    theta[0,3] = thetaModel[6] # k6
    theta[1,3] = thetaModel[7] # k5
    theta[2,1] = thetaModel[8] # K8
    theta[2,2] = thetaModel[9] # K9
    theta[2,3] = thetaModel[10] # K10
    theta[1,1] = thetaModel[11] # km2
    theta[0,2] = thetaModel[12] # k3
    return theta
