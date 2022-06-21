#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import KinaseModel as KM
import Fit_ReactionVelocity
import Fit_DDXeffect
import Fit_DDXdose
import Fit_GenericModels
import getData

#######################################################################################################################

def ReactionVelocity_fit(runs, kinases=None, procedure='fit'):
    kinases = ['CK1', 'CK2'] if not kinases else [kinases]
    for kinase in kinases:
        Fit_ReactionVelocity.Fitting(runs, kinase, procedure)

#######################################################################################################################

def ReactionVelocity_profileLikelihoods(kinases=None):
    kinases = ['CK1', 'CK2'] if not kinases else [kinases]
    for kinase in kinases:
        print(kinase)
        Fit_ReactionVelocity.ProfileLikelihoods(kinase)

#######################################################################################################################

def DDXeffect_fit(runs, kinases=None, modeDDX=False, procedure='fit'):
    kinases = ['CK1', 'CK2'] if not kinases else [kinases]
    for kinase in kinases:
        print(kinase)
        if modeDDX:
            for ATP in ['00', '10', '01', '11']:
                for pepSub in ['00', '10', '01', '11']:
                    print(ATP, pepSub)
                    modeDDX = ATP + pepSub
                    Fit_DDXeffect.Fitting(runs, kinase, modeDDX, procedure=procedure)
        else:
            Fit_DDXeffect.Fitting(runs, kinase, procedure=procedure)

#######################################################################################################################

def DDXeffect_profileLikelihoods(kinases=None, modeDDX=None):
    kinases = ['CK1', 'CK2'] if not kinases else [kinases]
    for kinase in kinases:
        print(kinase)
        Fit_DDXeffect.ProfileLikelihoods(kinase, modeDDX)

#######################################################################################################################

def DDXdose_fit(runs, kinases=None):
    kinases = ['CK1', 'CK2'] if not kinases else [kinases]
    modeDDX = {'CK1': '1101', 'CK2': '0101'}
    for kinase in kinases:
        print(kinase)
        Fit_DDXdose.Fitting(runs, kinase, modeDDX[kinase])

#######################################################################################################################

def DDXdose_profileLikelihoods(runs, kinases=None):
    kinases = ['CK1', 'CK2'] if not kinases else [kinases]
    modeDDX = {'CK1': '1101', 'CK2': '0101'}
    for kinase in kinases:
        print(kinase)
        Fit_DDXdose.Fitting(runs, kinase, modeDDX[kinase])

#######################################################################################################################

def GenericModels_fit(runs, kinase=None, exp=None):
    kinases = ['CK1', 'CK2'] if not kinase else [kinase]
    exps = ['ATP', 'pepSub', 'ADP', 'pepMut'] if not exp else [exp]
    for kinase in kinases:
        for exp in exps:
            print(kinase, exp)
            Fit_GenericModels.Fitting(runs, kinase, exp)

#######################################################################################################################
