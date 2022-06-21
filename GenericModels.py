#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def FunctionType(theta, dose, model):
    if model in ['ATP', 'pepSub']:
        return theta[0] * dose / (theta[1] + dose + dose**2/theta[2])
    if model in ['ADP', 'pepMut']:
        return theta[0] * theta[1] / (theta[1] + dose)
