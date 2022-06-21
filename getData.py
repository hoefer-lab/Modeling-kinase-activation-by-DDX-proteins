#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def ReactionVelocity(kinase):
    path = 'Data/'
    facADP_kinase = {'CK1': 1.0, 'CK2': 1.0}
    facMut_kinase = {'CK1': 0.02, 'CK2': 0.1}
    data = {}
    xl = pd.ExcelFile(path + kinase + '_basic.xlsx')
    for exp in xl.sheet_names:
        data_raw = xl.parse(sheet_name=exp, index_col=None, header=None, dtype=np.float64)
        data_raw = data_raw.values
        facADP = facADP_kinase[kinase] if exp == 'ADP' else 0.0
        facMut = facMut_kinase[kinase] if exp == 'pepMut' else 0.0
        data_exp = {'kinase': data_raw[0,0], 'ATP': data_raw[0,1:], 'pepSub': data_raw[1:,0], 'factor_ADP': facADP, 'factor_pepMut': facMut, 'values': data_raw[1:,1:]}
        data.update({exp: data_exp})
    return data

#######################################################################################################################

def ReactionVelocity_DDX(kinase, fold=False, dose=False):
    path = 'Data/' + kinase + '_DDX'
    if dose: path += 'dose'
    data = {}
    xl = pd.ExcelFile(path + '.xlsx')
    for exp in xl.sheet_names:
        data_raw = xl.parse(sheet_name=exp, index_col=None, header=0, dtype=np.float64)
        data_raw[data_raw < 0] = np.nan
        data_exp = {}
        data_exp['kinase'] = data_raw['kinase'].iloc[0]
        data_exp['ATP'] = data_raw['ATP'].values
        data_exp['pepSub'] = data_raw['pepSub'].values
        data_exp['ADP'] = data_raw['ADP'].values
        data_exp['pepMut'] = data_raw['pepMut'].values
        data_exp['DDX'] = data_raw['DDX'].values
        if fold:
            data_exp['values'] = data_raw.loc[:,'DDX3':'DDX56'].values / data_raw['ctrl'].values[:,np.newaxis]
        else:
            data_exp['values'] = data_raw.loc[:,'ctrl':'DDX56'].values
        data.update({exp: data_exp})
    return data

#######################################################################################################################

def ReactionVelocity_DDXctrl(kinase):
    path = 'Data/' + kinase + '_DDXdose'
    ctrls = ['ctrl', 'GFP', 'BSA', 'g-glob', 'DDX27']
    xl = pd.ExcelFile(path + '.xlsx')
    data = xl.parse(index_col=None, header=0, dtype=np.float64)
    data[data < 0] = 0.0
    data[ctrls] = data[ctrls].div(data['ctrl'], axis=0)
    return data

#######################################################################################################################
