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

#senoidal 2kHz
tt, xx = mi_funcion_sen(vmax=1, dc=0, ff=2000, ph=0, N=1000, fs=400000)

#senoidal 2kHz - amplificada y desfasada en π/2
tt, yy = mi_funcion_sen(vmax=2, dc=0, ff=2000, ph=(np.pi)/2, N=1000, fs=400000)

#senoidal 1kHz moduladora
tt, mm = mi_funcion_sen(vmax=1, dc=0, ff=1000, ph=0, N=1000, fs=400000)

#modulacion
nn = xx * mm

N=1000 
energia_a = (1/N)*(np.sum(np.abs(xx)**2))
print('la potencia de a es =' , energia_a)
# print(threshold)
#A²/2

energia_b = (1/N)*(np.sum(np.abs(yy)**2))
print('la potencia de b es =' , energia_b)

energia_c = (1/N)*(np.sum(np.abs(nn)**2))
print('la potencia de c es =' , energia_c)


#recorte en amplitud (clipping)
a = 1
a_clipped = a*0.75
clipped = np.clip(xx, -a_clipped, a_clipped)

energia_d = (1/N)*(np.sum(np.abs(clipped)**2))
print('la potencia de d es =' , energia_d)

#Señal cuadrada
def cuadrada(vmax=1, dc=0, ff=2000, ph=0, N=1000, fs=400000):
        Ts = 1/fs
        tc = np.arange(0, N*Ts, Ts)
        cc = vmax * np.sign(np.sin(2 * np.pi * ff * tc + ph) + dc)
        return tc, cc
        
tc, cc = cuadrada(vmax=1, dc=0, ff=4000, ph=0, N=1000, fs=400000)

energia_e = (1/N)*(np.sum(np.abs(cc)**2))
print('la potencia de e es =' , energia_e)


#Pulso de 10ms
fs = 1000
Ts = 1/fs
P = 1000
tp = np.arange(0, P*Ts, Ts)
X = np.zeros(P, dtype = complex)
start = 200
X [start:start+10] = 1



energia_f = (1/P)*(np.sum(np.abs(tp)**2))
print('la potencia de f es =' , energia_f)


def ortogonalidad(señal1, señal2, nombre1, nombre2, tolerancia=1e-10):
    producto_interno = np.sum(señal1 * señal2)
    if (producto_interno) < tolerancia:
        print(f"La señal '{nombre1}' y la señal '{nombre2}' son ortogonales. Producto interno = {producto_interno:.2e}")
    else:
        print(f"La señal '{nombre1}' y la señal '{nombre2}' NO son ortogonales. Producto interno = {producto_interno:.2e}")

ortogonalidad(xx, yy, 'a', 'b')
ortogonalidad(xx, nn, 'a', 'c')
ortogonalidad(xx, clipped, 'a', 'd')
ortogonalidad(xx, cc, 'a', 'e')
ortogonalidad(xx, X, 'a', 'f')

# producto_interno = np.sum(xx * X)
# if producto_interno == 0:
#     print('Las señales a y f son ortogonales')
# else:
#     print('Las señales a y f no son ortogonales. Producto interno =', producto_interno)
# #VER

#Autocorrelacion
autocorrelacion = np.correlate(xx, xx, mode='full')
cruzada_1 = np.correlate(xx, yy, mode = 'full')
cruzada_2 = np.correlate(xx, nn, mode = 'full')
cruzada_3 = np.correlate(xx, clipped, mode = 'full')
cruzada_4 = np.correlate(xx, cc , mode = 'full')
cruzada_5 = np.correlate(xx, X, mode = 'full')


# Graficar

plt.plot(tt, xx, label= 'ff=2kHz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')
plt.title('Señales Senoidales')
plt.grid(True)
plt.plot(tt,yy, label= 'ff=2kHz, desfasada en pi/2')
plt.legend()

plt.figure(2)
plt.grid(True)
plt.plot(tt, nn)
plt.title('Señal Modulada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')

fig, axs = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)  

axs[0, 0].plot(autocorrelacion)
axs[0, 0].set_title('Autocorrelación - señal senoidal original')
axs[0, 0].grid(True)

axs[1, 0].plot(cruzada_1)  
axs[1, 0].set_title('Correlacion cruzada entre señales a y b' )
axs[1, 0].grid(True)

axs[2, 0].plot(cruzada_2)  
axs[2, 0].set_title('Correlacion cruzada entre a y c')
axs[2, 0].grid(True)

axs[0, 1].plot(cruzada_3) 
axs[0, 1].set_title('Correlacion cruzada entre a y d')
axs[0, 1].grid(True)

axs[1, 1].plot(cruzada_4)  
axs[1, 1].set_title('Correlacion cruzada entre a y e')
axs[1, 1].grid(True)

axs[2, 1].plot(cruzada_5)  
axs[2, 1].set_title('Correlacion cruzada entre a y f')
axs[2, 1].grid(True)


plt.figure(4)
plt.grid(True)
plt.plot(tc,cc)
plt.title('Señal Cuadrada 4kHz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')

plt.figure(5)
plt.grid(True)
plt.plot(tt, clipped)
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')
plt.title('Señal recortada en Amplitud al 75%')

plt.show()

plt.figure(6)
plt.grid(True)
plt.plot(tp, X)
plt.title('Pulso de 10ms')
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')
plt.show








