#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib

matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.size'] = 7
matplotlib.rcParams['axes.linewidth'] = 0.5
matplotlib.rcParams['axes.titlepad'] = 2
matplotlib.rcParams['axes.titlesize'] = 'medium'
matplotlib.rcParams['grid.linewidth'] = 0.5
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 1.5
matplotlib.rcParams['xtick.major.width'] = 0.5
matplotlib.rcParams['xtick.major.pad'] = 1
matplotlib.rcParams['xtick.minor.width'] = 0.3
matplotlib.rcParams['ytick.major.width'] = 0.5
matplotlib.rcParams['ytick.major.pad'] = 1
matplotlib.rcParams['ytick.minor.width'] = 0.3
matplotlib.rcParams['savefig.format'] = 'svg'
matplotlib.rc('text.latex', preamble=r'\usepackage{cmbright}')


def globalLabel(ax, axes=None, arrowPos=0, arrowStart=None, arrowLen=1, direction='pos', text=None, textOffset=0, width=0.5, **kwargs):
    if axes not in ['x', 'y']:
        raise Exception("Missing axes argument in call of globalLabel. Set either to 'x' or 'y'.")
    if arrowStart is None:
        arrowStart = (1 - arrowLen) / 2
    if direction == 'pos':
        pass
    elif direction == 'neg':
        arrowStart, arrowLen = arrowStart+arrowLen, -arrowLen
    else:
        raise Exception("Unknown direction specified in call of globalLabel. Options are 'pos' (default) and 'neg'.")

    if axes == 'x':
        ax.annotate('', xy=(arrowStart+arrowLen, arrowPos), xytext=(arrowStart, arrowPos), annotation_clip=False, arrowprops=dict(facecolor='black', lw=width, arrowstyle="->"))
        ax.text(arrowStart+0.5*arrowLen, arrowPos+textOffset, text, ha='center', va='top', **kwargs)
    elif axes == 'y':
        ax.annotate('', xy=(arrowPos, arrowStart+arrowLen), xytext=(arrowPos, arrowStart), annotation_clip=False, arrowprops=dict(facecolor='black', lw=width, arrowstyle="->"))
        ax.text(arrowPos+textOffset, arrowStart+0.5*arrowLen, text, va='center', ha='right', rotation='vertical', **kwargs)
