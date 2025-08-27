#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 22:44:20 2025

@author: victoria24
"""

import numpy as np
import matplotlib.pyplot as plt

def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, N=1000, fs=1000):
    Ts = 1/fs
    tt = np.arange(0, N*Ts, Ts)
    #print(tt)
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    return tt, xx

# Llamar a la función
tt, xx = mi_funcion_sen(vmax=1, dc=0, ff=20000, ph=0, N=1000, fs=400000)

tt, mm = mi_funcion_sen(vmax=1, dc=0, ff=1000, ph=0, N=1000, fs=400000)

nn = xx * (1.5+mm)


plt.grid(True)
plt.plot(tt, nn, label= 'modulada')






