#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import getData
import getFitResults
import PlotWrapper as PW
import KinaseModel as KM
import GenericModels as GM
import DDXimpact

'''
generate figures as in the manuscript
'''

#######################################################################################################################

def getParameters(kinase, modeDDX=None):
    theta = getFitResults.ReactionVelocity(kinase)[0]
    thetaBind = KM.setParameters(theta)
    if kinase == 'CK1': modeDDX = '1101'
    if kinase == 'CK2': modeDDX = '0101'
    thetaDDX, sigma = getFitResults.DDXeffect(kinase, modeDDX)[:2]
    if not modeDDX:
        EffSizeDDX = thetaDDX.reshape(4,-1)
    else:
        EffSizeDDX = np.ones((4,4))
        modeDDX = np.array([int(mode) for mode in modeDDX], dtype=bool)
        if (modeDDX[0] + modeDDX[1] > 0) and (modeDDX[2] + modeDDX[3] == 0):
            EffSizeDDX[modeDDX] = np.concatenate((thetaDDX, thetaDDX, np.ones(4), np.ones(4))).reshape(4,4)[modeDDX]
        if (modeDDX[0] + modeDDX[1] == 0) and (modeDDX[2] + modeDDX[3] > 0):
            EffSizeDDX[modeDDX] = np.concatenate((np.ones(4), np.ones(4), thetaDDX, thetaDDX)).reshape(4,4)[modeDDX]
        if (modeDDX[0] + modeDDX[1] > 0) and (modeDDX[2] + modeDDX[3] > 0):
            EffSizeDDX[modeDDX] = np.concatenate((thetaDDX[:4], thetaDDX[:4], thetaDDX[4:], thetaDDX[4:])).reshape(4,4)[modeDDX]
    return thetaBind, EffSizeDDX, sigma

#######################################################################################################################

def setConcentrations(data, experiment):
    E = data['kinase']
    A = np.geomspace(data['ATP'][0], data['ATP'][-1], 100)
    B = np.geomspace(data['pepSub'][0], data['pepSub'][-1], 100)
    Q = data['ADP'][1] > 0 and np.geomspace(data['ADP'][1], data['ADP'][-1], 100)
    M = data['pepMut'][1] > 0 and np.geomspace(data['pepMut'][1], data['pepMut'][-1], 100)
    return dict(kinase=E, ATP=A, pepSub=B, ADP=Q, pepMut=M)

#######################################################################################################################

def MS_Fig3b(saveOpt, view):
    kinases = ['CK1', 'CK2']

    fig, ax = plt.subplots(2, 4, figsize=(4.5,2.0))
    ax0 = fig.add_subplot(111)
    PW.globalLabel(ax0, axes='x', arrowPos=-0.12, text=r'Substrate ($\mu$M)', textOffset=-0.02)
    PW.globalLabel(ax0, axes='y', arrowPos=-0.055, text=r'v (nM/min)', textOffset=-0.004)
    ax0.axis('off')

    fig.subplots_adjust(hspace=0.4, wspace=0.35, top=0.93, bottom=0.18, left=0.08, right=0.95)

    for k, kinase in enumerate(kinases):
        if kinase == 'CK1':
            yticks = [np.linspace(0,20,3), np.linspace(0,100,3), np.linspace(0,200,3), np.linspace(0,300,3), np.linspace(0,300,3)]
        if kinase == 'CK2':
            yticks = [np.linspace(0,4,3), np.linspace(0,30,4), np.linspace(0,60,3), np.linspace(0,100,3), np.linspace(0,100,3)]
        ATPindex = np.array([1, 4, 6, 8])
        ATPvalues = ['0.78', '6.25', '25', '100']

        data = getData.ReactionVelocity(kinase)['basic']
        theta, sigma, MLE, LLmin = getFitResults.ReactionVelocity(kinase)[:4]
        theta = KM.setParameters(theta)
        for j, atpind in enumerate(ATPindex):
            pepSub = np.geomspace(data['pepSub'][0], data['pepSub'][-1], 100)
            ymodel = KM.ReactionVelocity(data['kinase'], data['ATP'][atpind], pepSub, 0, 0, theta)
            ax[k,j].fill_between(pepSub, (1-sigma)*ymodel, (1+sigma)*ymodel, color=((0.8,0.8,0.8)))
            ax[k,j].plot(data['pepSub'], data['values'][:,atpind], marker='o', linestyle='')
            ax[k,j].semilogx(pepSub, ymodel, color='k')
            ax[k,j].set_xticks(10**np.unique(np.floor(np.log10( data['pepSub'][data['pepSub']>1] ))))
            ax[k,j].set_ylim(bottom=0)
            ax[k,j].set_yticks(yticks[j])
            ax[k,j].tick_params(axis='x', which='both', bottom=True, direction='out', zorder=10, pad=-1)
            ax[k,j].tick_params(axis='y', which='major', pad=1)
            if k == 0:
                ax[k,j].set_title(ATPvalues[j] + r'$\mu$M', fontsize='medium', pad=2)
    if saveOpt.save:
        plt.savefig('Figures/Fig3b' + '.' + saveOpt.format, dpi=saveOpt.dpi)
    if view: fig.show()

#######################################################################################################################

def MS_Fig3cdfg(kinase, plotType, saveOpt, view):
    fig, ax = plt.subplots(1, 2, figsize=(2.1,1.0), sharey=True)
    fig.subplots_adjust(hspace=0.2, wspace=0.25, top=0.92, bottom=0.30, left=0.15, right=0.95)
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    data = getData.ReactionVelocity_DDX(kinase)
    data = {exp: data[exp] for exp in ['ATP', 'pepSub']}
    thetaBind, EffSizeDDX, sigma = getParameters(kinase)
    theta = DDXimpact.setDDXimpact(thetaBind, EffSizeDDX)
    for j, (exp, dd) in enumerate(data.items()):
        Conc = setConcentrations(dd, exp)
        E = Conc['kinase']
        A = Conc['ATP'][:, np.newaxis] if exp == 'ATP' else Conc['ATP'][0]
        B = Conc['pepSub'][:, np.newaxis] if exp == 'pepSub' else Conc['pepSub'][0]
        v = KM.ReactionVelocity(E, A, B, 0, 0, theta)
        vModel = v / np.amax(v)
        thetaPheno = getFitResults.PhenomenologicalDDX(kinase, exp)
        vDataPheno = GM.FunctionType(thetaPheno[:,:,np.newaxis], Conc[exp], exp)
        vmax = np.amax(vDataPheno)
        vDataPheno /= vmax
        concData = dd[exp]
        vData = dd['values'] / vmax
        for d in range(5):
            if plotType == 'data':
                ax[j].semilogx(concData, vData[:,d], marker='o', linestyle='', color=cycle[d])
                ax[j].semilogx(Conc[exp], vDataPheno[d], color=cycle[d])
            if plotType == 'model':
                ax[j].semilogx(Conc[exp], vModel[:,d])

        ax[j].set_ylim(bottom=0, top=1.2 )
        ax[j].set_xlim(left=np.min([1,ax[j].get_xlim()[0]]), right=np.max([1e3, ax[j].get_xlim()[-1]]) )
        ax[j].xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
        ax[j].yaxis.set_major_locator(ticker.FixedLocator([0, 0.5, 1]))
        ax[j].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax[j].tick_params(axis='x', which='major', bottom=True, direction='out', pad=-1)
        ax[j].tick_params(axis='y', which='major', pad=0)
        xlabel = exp
        if exp == 'pepSub': xlabel = 'Substrate'
        ax[j].set_xlabel(xlabel + ' ($\mu$M)', labelpad=1)
        if j == 0:
            ax[j].set_ylabel(r'v / v$_\mathrm{max}$', labelpad=1)
    if saveOpt.save:
        figPanel = 'cd' if plotType == 'data' else 'fg'
        figPanel = figPanel[0] if kinase == 'CK1' else figPanel[1]
        plt.savefig('Figures/Fig3' + figPanel + '.' + saveOpt.format, dpi=saveOpt.dpi)
    if view: fig.show()

#######################################################################################################################

def MS_Fig4ab(exp, saveOpt, view):
    xlabel = dict(ADP = r'ADP', pepMut = r'Mutant peptide')
    ylabel = dict(ADP = r'v(ADP) / v(ADP=0)', pepMut = r'v(Mut) / v(Mut=0)')
    fig, ax = plt.subplots(2, 2, figsize=(2.0,2.0), sharey=True)
    ax0 = fig.add_subplot(111)
    PW.globalLabel(ax0, axes='x', arrowPos=-0.12, text=xlabel[exp] + ' ($\mu$M)', textOffset=-0.02)
    PW.globalLabel(ax0, axes='y', arrowPos=-0.15, text=ylabel[exp], textOffset=-0.01)
    ax0.axis('off')
    fig.subplots_adjust(hspace=0.4, wspace=0.15, top=0.92, bottom=0.18, left=0.2, right=0.97)
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for j, kinase in enumerate(['CK1', 'CK2']):
        dd = getData.ReactionVelocity_DDX(kinase)[exp]
        thetaBind, EffSizeDDX, sigma = getParameters(kinase)
        theta = DDXimpact.setDDXimpact(thetaBind, EffSizeDDX)
        Conc = setConcentrations(dd, exp)
        E = Conc['kinase']
        A, Q = (Conc['ATP'][:, np.newaxis], Conc['ADP'][:, np.newaxis]) if exp == 'ADP' else (Conc['ATP'][0], 0)
        B, M = (Conc['pepSub'][:, np.newaxis], Conc['pepMut'][:, np.newaxis]) if exp == 'pepMut' else (Conc['pepSub'][0], 0)
        v0 = KM.ReactionVelocity(E, A, B, 0, 0, theta)
        vModel = KM.ReactionVelocity(E, A, B, Q, M, theta) / v0
        thetaPheno = getFitResults.PhenomenologicalDDX(kinase, exp)
        vDataPheno = GM.FunctionType(thetaPheno[:,:,np.newaxis], Conc[exp], exp)
        vmax = np.amax(vDataPheno, axis=1)[:, np.newaxis]
        vDataPheno /= vmax
        concData = dd[exp]
        vData = dd['values'] / vmax.T
        for k, plotType in enumerate(['data', 'model']):
            for d in range(5):
                if plotType == 'data':
                    ax[k,j].semilogx(concData, vData[:,d], marker='o', linestyle='', color=cycle[d])
                    ax[k,j].semilogx(Conc[exp], vDataPheno[d], color=cycle[d])
                if plotType == 'model':
                    ax[k,j].semilogx(Conc[exp], vModel[:,d])

            ax[k,j].set_ylim(bottom=0, top=1.4)
            ax[k,j].set_xlim(left=np.min([1,ax[k,j].get_xlim()[0]]), right=np.max([1e3, ax[k,j].get_xlim()[-1]]) )
            ax[k,j].xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
            ax[k,j].yaxis.set_major_locator(ticker.FixedLocator([0, 0.5, 1]))
            ax[k,j].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            ax[k,j].tick_params(axis='x', which='major', bottom=True, direction='out', pad=-1)
            ax[k,j].tick_params(axis='y', which='major', pad=0)
            ax[k,j].set_xticklabels(['', '', '$\mathregular{10^{-1}}$', '', '$\mathregular{10^1}$', '', '$\mathregular{10^3}$'])
    if saveOpt.save:
        figPanel = 'a' if exp == 'ADP' else 'b'
        plt.savefig('Figures/Fig4' + figPanel + '.' + saveOpt.format, dpi=saveOpt.dpi)
    if view: fig.show()

#######################################################################################################################

def MS_Fig4c(saveOpt, view):
    xlabel = r'DDX (nM)'
    ylabel = r'v(DDX) / v(DDX=0)'
    fig, ax = plt.subplots(2, 2, figsize=(2.2,2.0), sharey=False)
    ax0 = fig.add_subplot(111)
    PW.globalLabel(ax0, axes='x', arrowPos=-0.12, text=xlabel, textOffset=-0.02)
    PW.globalLabel(ax0, axes='y', arrowPos=-0.12, text=ylabel, textOffset=-0.01)
    ax0.axis('off')
    fig.subplots_adjust(hspace=0.4, wspace=0.25, top=0.92, bottom=0.18, left=0.16, right=0.97)
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    modeDDX = dict(CK1 = '1101', CK2 = '0101')
    for j, kinase in enumerate(['CK1', 'CK2']):
        dd = getData.ReactionVelocity_DDX(kinase, dose=True, fold=False)['Sheet1']
        thetaBind, EffSizeDDX, sigma = getParameters(kinase)
        theta0 = getFitResults.ReactionVelocity(kinase)[0]
        theta0 = KM.setParameters(theta0)
        KDDX, delRate, sigma = getFitResults.DDXdose(kinase, modeDDX[kinase])[:3]
        DDXdose = np.geomspace(dd['DDX'][0], dd['DDX'][-1], 100)
        KDDX = np.concatenate(([0], KDDX))
        for k, plotType in enumerate(['data', 'model']):
            for d in range(5):
                facDDX = np.tile( ( 1 + delRate*KDDX[d]*DDXdose ) / ( 1 + KDDX[d]*DDXdose ), 4).reshape(4,-1)
                EffSizeDDX = np.ones_like(facDDX)
                mode = np.array([int(mode) for mode in modeDDX[kinase]], dtype=bool)
                EffSizeDDX[mode] = facDDX[mode]
                theta = DDXimpact.setDDXimpact(theta0, EffSizeDDX)
                v = KM.ReactionVelocity(dd['kinase'], dd['ATP'][0], dd['pepSub'][0], 0, 0, theta)
                if plotType == 'data':
                    ax[k,j].semilogx(dd['DDX'], dd['values'][:,d] / dd['values'][:,0], marker='o', linestyle='', color=cycle[d])
                ax[k,j].semilogx(DDXdose, v[1:] / v[0], color=cycle[d])
            ax[k,j].set_ylim(bottom=0, top=12 if kinase=='CK1' else 6)
            ax[k,j].set_xlim(left=np.min([1,ax[k,j].get_xlim()[0]]), right=np.max([1e3, ax[k,j].get_xlim()[-1]]) )
            ax[k,j].xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
            ax[k,j].yaxis.set_major_locator(ticker.FixedLocator([0, 5, 10] if kinase=='CK1' else [0, 2, 4, 6]))
            ax[k,j].yaxis.set_minor_locator(ticker.MultipleLocator(1))
            if kinase == 'CK2':
                ax[k,j].set_xticklabels(['', '', '$\mathregular{10^{-1}}$',  '', '$\mathregular{10^1}$', '', '$\mathregular{10^3}$'])
            ax[k,j].tick_params(axis='x', which='major', bottom=True, direction='out', pad=-1)
            ax[k,j].tick_params(axis='y', which='major', pad=0)
    if saveOpt.save:
        plt.savefig('Figures/Fig4c.' + saveOpt.format, dpi=saveOpt.dpi)
    if view: fig.show()

#######################################################################################################################

def Supp_Fig78(kinase, saveOpt, view): # Fits reaction velocities

    def setUpperLim(lim):
        if np.floor(np.log10(lim)) < 2:
            return int( np.ceil(10 * lim / 10**np.ceil(np.log10(lim))) * 10**np.floor(np.log10(lim)) )
        else:
            return int( np.ceil(10 * lim / 10**np.ceil(np.log10(lim) - 1)) * 10**np.floor(np.log10(lim) - 1) )

    fig, ax = plt.subplots(4, 6, figsize=(7,4), sharex=False)
    ax0 = fig.add_subplot(111)
    PW.globalLabel(ax0, axes='x', arrowPos=-0.07, text=r'Substrate ($\mu$M)', textOffset=-0.01)
    PW.globalLabel(ax0, axes='y', arrowPos=-0.05, text=r'v (nM/min)', textOffset=-0.004)
    ax0.axis('off')
    fig.subplots_adjust(hspace=0.62, wspace=0.37, top=0.9, bottom=0.1, left=0.07, right=0.95)

    data = getData.ReactionVelocity(kinase)
    theta, sigma = getFitResults.ReactionVelocity(kinase)[:2]
    theta = KM.setParameters(theta)
    count = 0
    for exp, dd in data.items():
        for atpind, atp in enumerate(dd['ATP']):
            pepSub = np.geomspace(dd['pepSub'][0], dd['pepSub'][-1], 100)
            ymodel = KM.ReactionVelocity(dd['kinase'], atp, pepSub, dd['factor_ADP']*atp, dd['factor_pepMut']*pepSub, theta)
            row = count // 6
            col = count % 6
            count += 1
            ax[row,col].fill_between(pepSub, (1-sigma)*ymodel, (1+sigma)*ymodel, color=((0.8,0.8,0.8)))
            ax[row,col].semilogx(dd['pepSub'], dd['values'][:,atpind], marker='o', linestyle='')
            ax[row,col].semilogx(pepSub, ymodel, color='k')
            ax[row,col].set_ylim(bottom=0, top=setUpperLim(ax[row,col].get_ylim()[-1]) )
            ax[row,col].yaxis.set_major_locator(ticker.LinearLocator(numticks=3))
            ax[row,col].tick_params(axis='y', which='major', pad=1)
            if kinase == 'CK1':
                ax[row,col].set_xticks([1e1, 1e2, 1e3])
            if kinase == 'CK2':
                ax[row,col].set_xticks([1e0, 1e1, 1e2])
            ax[row,col].tick_params(axis='x', which='major', bottom=True, direction='out', pad=-1)
            ax[row,col].tick_params(axis='x', which='minor', bottom=False)
            atp = np.round(100 * atp) / 100
            if atp > 20:
                atp = int(atp)
            ax[row,col].set_title('ATP = ' + str(atp) + r'$\mu$M', fontsize='medium', pad=2)
    if saveOpt.save:
        figNum = '7' if kinase == 'CK1' else '8'
        plt.savefig('Figures/SuppFig' + figNum + '.' + saveOpt.format, dpi=saveOpt.dpi)
    if view: fig.show()

########################################################################################################################

def Supp_Fig9ac(kinase, saveOpt, view): # Fits DDX effect
    fig, ax = plt.subplots(2, 4, figsize=(4.1,1.8), sharex=False, sharey=True)
    ax0 = fig.add_subplot(111)
    PW.globalLabel(ax0, axes='y', arrowPos=-0.08, text=r'v(DDX) / v(DDX=0)', textOffset=-0.004)
    PW.globalLabel(ax0, axes='x', arrowPos=0.49, text=r'ATP ($\mu$M)', textOffset=-0.02)
    PW.globalLabel(ax0, axes='x', arrowPos=-0.14, text=r'Substrate ($\mu$M)', textOffset=-0.02)
    ax0.axis('off')
    fig.subplots_adjust(hspace=0.7, wspace=0.25, top=0.9, bottom=0.18, left=0.13, right=0.95)
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    DDX = ['DDX3', 'DDX5', 'DDX27', 'DDX56']
    data = getData.ReactionVelocity_DDX(kinase, fold=True)
    data = {exp: data[exp] for exp in ['ATP', 'pepSub']}
    thetaBind, EffSizeDDX, sigma = getParameters(kinase)
    theta = DDXimpact.setDDXimpact(thetaBind, EffSizeDDX)
    for j, (exp, dd) in enumerate(data.items()):
        # data
        Conc = setConcentrations(dd, exp)
        E = Conc['kinase']
        A = Conc['ATP'][:, np.newaxis] if exp == 'ATP' else Conc['ATP'][0]
        B = Conc['pepSub'][:, np.newaxis] if exp == 'pepSub' else Conc['pepSub'][0]
        v = KM.ReactionVelocity(E, A, B, 0, 0, theta)
        vModel = v[:,1:] / v[:,0][:,np.newaxis]
        concData = dd[exp]
        vData = dd['values']
        for d in range(4):
            ax[j,d].fill_between(Conc[exp], vModel[:,d]-sigma, vModel[:,d]+sigma, color=((0.8,0.8,0.8)))
            ax[j,d].semilogx(concData, vData[:,d], marker='o', linestyle='', color=cycle[d+1])
            ax[j,d].semilogx(Conc[exp], vModel[:,d], 'k')
            ax[j,d].set_ylim(bottom=0, top=20 if kinase == 'CK1' else 5)
            ax[j,d].yaxis.set_major_locator(ticker.LinearLocator(numticks=3))
            ax[j,d].yaxis.set_minor_locator(ticker.LinearLocator(numticks=11))
            ax[j,d].set_xlim(left=np.min([1,ax[j,d].get_xlim()[0]]), right=np.max([1e3, ax[j,d].get_xlim()[-1]]) )
            ax[j,d].xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
            ax[j,d].tick_params(axis='x', which='major', bottom=True, direction='out', pad=-1)
            ax[j,d].tick_params(axis='y', which='major', pad=1)
            if j==0:
                ax[j,d].set_title(DDX[d], fontsize='medium', pad=2)
    if saveOpt.save:
        figPanel = 'a' if kinase == 'CK1' else 'c'
        plt.savefig('Figures/SuppFig9' + figPanel + '.' + saveOpt.format, dpi=saveOpt.dpi)
    if view: fig.show()

########################################################################################################################

def Supp_Fig9e(saveOpt, view): # Controls DDX dose
    fig, ax = plt.subplots(2, 1, figsize=(1.2,1.6))#, sharex=False, sharey=True)
    ax0 = fig.add_subplot(111)
    PW.globalLabel(ax0, axes='y', arrowPos=-0.25, text=r'v(DDX) / v(DDX=0)', textOffset=-0.015)
    ax0.axis('off')
    fig.subplots_adjust(hspace=0.5, wspace=0.25, top=0.92, bottom=0.2, left=0.3, right=0.94)
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ctrls = ['ctrl', 'GFP', 'BSA', 'g-glob', 'DDX27']
    colors = dict(zip(ctrls, [cycle[0], 'xkcd:jungle green', 'xkcd:dirty yellow', 'xkcd:darkish purple', cycle[3]]))
    modeDDX = dict(CK1 = '1101', CK2 = '0101')
    for k, kinase in enumerate(['CK1', 'CK2']):
        data = getData.ReactionVelocity_DDXctrl(kinase)
        thetaBind, EffSizeDDX, sigma = getParameters(kinase)
        theta0 = getFitResults.ReactionVelocity(kinase)[0]
        theta0 = KM.setParameters(theta0)
        KDDX, delRate, sigma = getFitResults.DDXdose(kinase, modeDDX[kinase])[:3]
        for ctrl in ['ctrl', 'GFP', 'BSA', 'g-glob', 'DDX27']:
            if ctrl == 'DDX27':
                DDXdose = np.geomspace(data['DDX'].iloc[0], data['DDX'].iloc[-1], 100)
                facDDX = np.tile( ( 1 + delRate*KDDX[2]*DDXdose ) / ( 1 + KDDX[2]*DDXdose ), 4).reshape(4,-1)
                EffSizeDDX = np.ones_like(facDDX)
                mode = np.array([int(mode) for mode in modeDDX[kinase]], dtype=bool)
                EffSizeDDX[mode] = facDDX[mode]
                theta = DDXimpact.setDDXimpact(theta0, EffSizeDDX)
                v = KM.ReactionVelocity(data['kinase'][0], data['ATP'][0], data['pepSub'][0], 0, 0, theta)
                ax[k].semilogx(data['DDX'], data[ctrl], marker='o', linestyle='', color=colors[ctrl])
                ax[k].semilogx(DDXdose, v[1:] / v[0], color=colors[ctrl])
            else:
                ax[k].semilogx(data['DDX'], data[ctrl], marker='o', linestyle='--', lw='0.2', color=colors[ctrl])

        ax[k].set_ylim(bottom=-0.5, top=12 if kinase=='CK1' else 6)
        ax[k].set_xlim(left=np.min([1,ax[k].get_xlim()[0]]), right=np.max([1e3, ax[k].get_xlim()[-1]]) )
        ax[k].xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
        ax[k].yaxis.set_major_locator(ticker.FixedLocator([0, 5, 10] if kinase=='CK1' else [0, 2, 4, 6]))
        ax[k].yaxis.set_minor_locator(ticker.MultipleLocator(1))
        if kinase == 'CK2':
            ax[k].set_xticklabels(['', '', '$\mathregular{10^{-1}}$',  '', '$\mathregular{10^1}$', '', '$\mathregular{10^3}$'])
        ax[k].tick_params(axis='x', which='major', bottom=True, direction='out', pad=-1)
        ax[k].tick_params(axis='y', which='major', pad=0)
        if k==1: ax[k].set_xlabel(r'DDX (nM)', labelpad=1)

    if saveOpt.save:
        plt.savefig('Figures/SuppFig9e.' + saveOpt.format, dpi=saveOpt.dpi)
    if view: fig.show()

#######################################################################################################################

def Supp_Fig9f(saveOpt, view): # Fits DDX dose
    fig, ax = plt.subplots(2, 4, figsize=(4.1,1.6), sharex='row', sharey='row')
    ax0 = fig.add_subplot(111)
    PW.globalLabel(ax0, axes='x', arrowPos=-0.17, text=r'DDX (nM)', textOffset=-0.02)
    PW.globalLabel(ax0, axes='y', arrowPos=-0.08, text=r'v(DDX) / v(DDX=0)', textOffset=-0.004)
    ax0.axis('off')
    fig.subplots_adjust(hspace=0.5, wspace=0.25, top=0.92, bottom=0.2, left=0.13, right=0.95)
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    DDX = ['DDX3', 'DDX5', 'DDX27', 'DDX56']
    for k, kinase in enumerate(['CK1', 'CK2']):
        dd = getData.ReactionVelocity_DDX(kinase, dose=True, fold=True)['Sheet1']
        theta0 = getFitResults.ReactionVelocity(kinase)[0]
        theta0 = KM.setParameters(theta0)
        modeDDX, KDDX, delRate, sigma = getFitResults.DDXdose(kinase)[:4]
        DDXeffect = DDXimpact.setDDXeffect(modeDDX)
        DDXdose = np.geomspace(dd['DDX'][0], dd['DDX'][-1], 100)
        theta = DDXimpact.setDDXimpact(theta0, np.ones(len(DDXdose)), np.ones(len(DDXdose)), DDXeffect)
        v0 = KM.ReactionVelocity(dd['kinase'], dd['ATP'][0], dd['pepSub'][0], 0, 0, theta)
        for j, ddx in enumerate(DDX):
            facDDX = ( 1 + delRate*KDDX[j]*DDXdose ) / ( 1 + KDDX[j]*DDXdose )
            theta = DDXimpact.setDDXimpact(theta0, facDDX, facDDX, DDXeffect)
            vModel = KM.ReactionVelocity(dd['kinase'], dd['ATP'][0], dd['pepSub'][0], 0, 0, theta)[1:] / v0[1:]
            ax[k,j].fill_between(DDXdose, vModel-sigma, vModel+sigma, color=((0.8,0.8,0.8)))
            ax[k,j].semilogx(dd['DDX'], dd['values'][:,j], marker='o', linestyle='', color=cycle[j+1])
            ax[k,j].semilogx(DDXdose, vModel, color='k')
            ax[k,j].set_ylim(bottom=0, top=12 if kinase == 'CK1' else 6)
            ax[k,j].yaxis.set_major_locator(ticker.LinearLocator(numticks=3))
            ax[k,j].yaxis.set_minor_locator(ticker.LinearLocator(numticks=11))
            ax[k,j].set_xlim(left=np.min([1,ax[k,j].get_xlim()[0]]), right=np.max([1e3, ax[k,j].get_xlim()[-1]]) )
            ax[k,j].xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
            if kinase == 'CK2':
                ax[k,j].set_xticklabels(['', '', '$\mathregular{10^{-1}}$',  '', '$\mathregular{10^1}$', '', '$\mathregular{10^3}$'])
            ax[k,j].tick_params(axis='x', which='major', bottom=True, direction='out', pad=-1)
            ax[k,j].tick_params(axis='y', which='major', pad=1)
            if k == 0: ax[k,j].set_title(ddx, fontsize='medium', pad=2)
    if saveOpt.save:
        plt.savefig('Figures/SuppFig9f.' + saveOpt.format, dpi=saveOpt.dpi)
    if view: fig.show()

#######################################################################################################################

def MS_Fig3(panel, saveOpt, view):
    if (not panel or panel == 'b'): MS_Fig3b(saveOpt, view)
    if (not panel or panel == 'c'): MS_Fig3cdfg('CK1', 'data', saveOpt, view)
    if (not panel or panel == 'd'): MS_Fig3cdfg('CK2', 'data', saveOpt, view)
    if (not panel or panel == 'f'): MS_Fig3cdfg('CK1', 'model', saveOpt, view)
    if (not panel or panel == 'g'): MS_Fig3cdfg('CK2', 'model', saveOpt, view)

#######################################################################################################################

def MS_Fig4(panel, saveOpt, view):
    if (not panel or panel == 'a'): MS_Fig4ab('ADP', saveOpt, view)
    if (not panel or panel == 'b'): MS_Fig4ab('pepMut', saveOpt, view)
    if (not panel or panel == 'c'): MS_Fig4c(saveOpt, view)

#######################################################################################################################

def Supp_Fig9(panel, saveOpt, view):
    if (not panel or panel == 'a'): Supp_Fig9ac('CK1', saveOpt, view)
    if (not panel or panel == 'c'): Supp_Fig9ac('CK2', saveOpt, view)
    if (not panel or panel == 'e'): Supp_Fig9e(saveOpt, view)
    if (not panel or panel == 'f'): Supp_Fig9f(saveOpt, view)

#######################################################################################################################

class saveOptions:
    save = False
    format = 'svg'
    dpi = 600

#######################################################################################################################

if __name__ == '__main__':
    import sys
    import os
    figure, panel, view = None, None, False
    saveOpt = saveOptions()
    while len(sys.argv) > 1:
        val = sys.argv.pop()
        if val in ['3', '4', '7', '8', '9']:
            figure = val
        if val in ['a', 'b', 'c', 'd', 'e', 'f']:
            panel = val
        if val in ['v', 'xv']:
            view = val == 'v'
        if val in ['s', 'save']:
            saveOpt.save = True
        if val[:6] == 'format':
            saveOpt.format = val[6:]
        if val[:3] == 'dpi':
            saveOpt.dpi = int(val[3:])
    if saveOpt.save and not os.path.isdir('Figures/'):
        os.mkdir('Figures/')
    if (not figure or figure == '3'):
        MS_Fig3(panel, saveOpt, view)
    if (not figure or figure == '4'):
        MS_Fig4(panel, saveOpt, view)
    if (not figure or figure == '7'):
        Supp_Fig78('CK1', saveOpt, view)
    if (not figure or figure == '8'):
        Supp_Fig78('CK2', saveOpt, view)
    if (not figure or figure == '9'):
        Supp_Fig9(panel, saveOpt, view)

    if view:
        input('press ENTER')
    plt.close('all')
