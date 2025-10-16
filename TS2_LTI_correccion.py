#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 00:40:37 2025

@author: victoria24
"""

import numpy as np
import matplotlib.pyplot as plt
import time as time

N = 3000  
fs = 1000    

def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, N=1000, fs=2):
    Ts = 1/fs
    tt = np.arange(0, N*Ts, Ts)
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    return tt, xx

tt, xx = mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, N=N, fs=fs)
tt, yy = mi_funcion_sen(vmax=2, dc=0, ff=2, ph=(np.pi)/2, N=N, fs=fs)

#modulacion
tt, mm = mi_funcion_sen(vmax=1, dc=0, ff=0.5, ph=0, N=N, fs=fs)
nn = xx * mm

#recortada en amplitud
a = 1
a_clipped = a*0.75
clipped = np.clip(xx, -a_clipped, a_clipped)

#cuadrada
def cuadrada(vmax=1, dc=0, ff=2000, ph=0, N=1000, fs=400000):
        Ts = 1/fs
        tc = np.arange(0, N*Ts, Ts)
        cc = vmax * np.sign(np.sin(2 * np.pi * ff * tc + ph) + dc)
        return tc, cc
tc, cc = cuadrada(vmax=1, dc=0, ff=1, ph=0, N=N, fs=fs)

#pulso cuadrado 10ms
duracion_ms = 10 
muestras_duracion = int((duracion_ms / 1000) * fs) 

pulso_cuadrado = np.zeros(N, dtype=float)
pulso_cuadrado[:muestras_duracion] = 1 

def calculo_potencia_promedio(señal):
    energia = np.sum(np.abs(señal)**2)
    potencia = (1/N)*energia 
    return potencia

def calculo_energia(señal):
    energia = np.sum(np.abs(señal)**2)
    return energia

start_time1 = time.time()

def respuesta(señal1):
    N_señal = len(señal1)
    Y = np.zeros(N_señal, dtype=float)
    for n in range(N_señal):
        if n==0:
            Y[n] = 3e-2*señal1[n] 
        elif n==1:
            Y[n] = 3e-2*señal1[n] + 5e-2*señal1[n-1] + 1.5*Y[n-1]
        else:     
            Y[n] = 3e-2*señal1[n] + 5e-2*señal1[n-1] + 3e-2*señal1[n-2] + 1.5*Y[n-1] - 0.5*Y[n-2]
    return Y
    
xx_salida = respuesta(xx)
end_time1 = time.time()

yy_salida = respuesta(yy)
nn_salida = respuesta(nn)
clipped_salida = respuesta(clipped)
cuadrada_salida = respuesta(cc)
pulso_c_salida = respuesta(pulso_cuadrado)


total_time1 = (end_time1 - start_time1)
print('tiempo total 1:', total_time1)
print('\n')

def plotear(entrada, respuesta, tiempo, title):
    plt.figure()
    plt.plot(tiempo, respuesta, label="señal de salida")
    plt.plot(tiempo, entrada, label="señal de entrada", color="red")
    plt.title(title)
    plt.grid(True)
    plt.xlabel('t [n]')
    plt.ylabel('x[n]/y[n]')
    plt.legend()
    plt.show()
    return 

plotear(xx, xx_salida, tt, 'Entrada Senoidal - Salida LTI')
potencia_salida_xx = calculo_potencia_promedio(xx_salida)
print(f"Potencia promedio de la señal de salida: {potencia_salida_xx:.2f}")

plotear(yy, yy_salida, tt, 'Entrada Senoidal Desplazada pi/2 - Salida LTI')
potencia_salida_yy = calculo_potencia_promedio(yy_salida)
print(f"Potencia promedio de la señal de salida: {potencia_salida_yy:.2f}")
      
plotear(nn, nn_salida, tt, 'Entrada Senoidal Modulada en Amp - Salida LTI')
potencia_salida_nn = calculo_potencia_promedio(nn_salida)
print(f"Potencia promedio de la señal de salida: {potencia_salida_xx:.2f}")    

plotear(clipped, clipped_salida, tt, 'Entrada Senoidal Clipped - Salida LTI')
potencia_salida_clipped = calculo_potencia_promedio(clipped_salida)
print(f"Potencia promedio de la señal de salida: {potencia_salida_clipped:.2f}")    

plotear(cc, cuadrada_salida, tc, 'Entrada Cuadrada - Salida LTI')
potencia_salida_cc = calculo_potencia_promedio(cuadrada_salida)
print(f"Potencia promedio de la señal de salida: {potencia_salida_cc:.2f}")

plotear(pulso_cuadrado, pulso_c_salida, tt, 'Entrada Senoidal Modulada en Amp - Salida LTI')
potencia_salida_pc = calculo_potencia_promedio(pulso_c_salida)
print(f"Potencia promedio de la señal de salida: {potencia_salida_pc:.2f}")    

#%%
#RESPUESTA AL IMPULSO 

#impulso
Ts = 1/fs
tp = np.arange(0, N*Ts, Ts)
P = np.zeros(N, dtype = float)
start = 0
P[start] = 1

#para ver el impulso
plt.figure(6)
plt.plot(tt, P, 'x')
plt.grid(True)
plt.xlim([-0.02, 0.02])
plt.xlabel('t[s]')
plt.ylabel('x[n]')
plt.title('Impulso')

start_time7 = time.time()
Z = np.zeros(N, dtype=float)
for n in range(N):
        if n==0:
            Z[n] = 3e-2*P[n] 
        elif n==1:
            Z[n] = 3e-2*P[n] + 5e-2*P[n-1] + 1.5*Z[n-1]
        else:     
            Z[n] = 3e-2*P[n] + 5e-2*P[n-1] + 3e-2*P[n-2] + 1.5*Z[n-1] - 0.5*Z[n-2]
            
plt.figure(7)
plt.plot(tt, Z, 'x')
plt.title("Respuesta al Impulso")
plt.grid(True)
plt.xlabel('t[n]')
plt.ylabel('h[n]')
plt.xlim([0, 10/fs]) #para que el limite sea en segundos
plt.show()

# end_time7 = time.time()
# total_time7 = (end_time7 - start_time7)
# print('tiempo de simulacion:', total_time7)
# potencia_h = calculo_potencia_promedio(Z)
# print("energia del impulso truncado:", energia_h)

#truncar rta al impulso
Lh = 1000
h = Z[:Lh]

#longitud de la salida N - Lh + 1
xx_valid = np.convolve(xx, h, 'valid')
len_xx_valid = len(xx_valid)
tt_valid = tt[:len_xx_valid]

# #%%
# # Convolución COMPLETA
# xx_conv_full = np.convolve(xx, h, 'full')

# # Truncar la convolución a la longitud de la entrada N (30000)
# xx_conv_N = xx_conv_full[:N] 

# # Ploteo de la salida por Ecuación en Diferencias (xx_salida) vs. Convolución (xx_conv_N)
# plt.figure()
# plt.plot(tt, xx_salida, label="Salida por Ecuación en Diferencias")
# plt.plot(tt, xx_conv_N, '--', label="Salida por Convolución (Truncada)")
# plt.title("Comparación de Salidas")
# plt.grid(True)
# plt.legend()
# plt.show()

#%%
yy_valid = np.convolve(yy, h, 'valid')
len_yy_valid = len(yy_valid)

nn_valid = np.convolve(nn, h, 'valid')
len_nn_valid = len(nn_valid)

clipped_valid = np.convolve(clipped, h, 'valid')
len_clipped_valid = len(clipped_valid)

cc_valid = np.convolve(cc, h, 'valid')
len_cc_valid = len(cc_valid)
tc_valid = tc[:len_cc_valid]

pulso_c_valid = np.convolve(pulso_cuadrado, h, 'valid')
len_pulso_c_valid = len(pulso_c_valid)

plt.subplot(3,2,1) 
plt.plot(tt_valid, xx_valid, 'x')
plt.plot()
plt.grid(True)
plt.xlabel('t [n]')
plt.ylabel('y[n]')
plt.title('Repuesta por convolucion - senoidal')

plt.subplot(3,2,2) 
plt.plot(tt_valid, yy_valid, 'x')
plt.plot()
plt.grid(True)
plt.xlabel('t [n]')
plt.ylabel('y[n]')
plt.title('Repuesta por convolucion - senoidal desplazada')

plt.subplot(3,2,3) 
plt.plot(tt_valid, nn_valid, 'x')
plt.plot()
plt.grid(True)
plt.xlabel('t [n]')
plt.ylabel('y[n]')
plt.title('Repuesta por convolucion - senoidal modulada')

plt.subplot(3,2,4) 
plt.plot(tt_valid, clipped_valid, 'x')
plt.plot()
plt.grid(True)
plt.xlabel('t [n]')
plt.ylabel('y[n]')
plt.title('senoidal recortada')

plt.subplot(3,2,5) 
plt.plot(tc_valid, cc_valid, 'x')
plt.plot()
plt.grid(True)
plt.xlabel('t [n]')
plt.ylabel('y[n]')
plt.title('Repuesta por convolucion - cuadrada')

plt.subplot(3,2,6) 
plt.plot(tt_valid, pulso_c_valid, 'x')
plt.plot()
plt.grid(True)
plt.xlabel('t [n]')
plt.ylabel('y[n]')
plt.title('Repuesta por convolucion - Pulso Cuadrado')

plt.tight_layout()
plt.show()


#%%

tt, sen_2 = mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, N=N, fs=fs)

def respuesta_2(señal, figure):
    A = np.zeros(N, dtype=float)
    for n in range(N):
        if n in range(10):
            A[n] = sen_2[n] #considero causal
        else:     
            A[n] = sen_2[n] + 3*sen_2[n-10]          
    plt.figure(figure)        
    plt.plot(tt, A, label='Señal de salida')
    plt.plot(tt, sen_2, label='Señal de entrada') 
    plt.grid(True)
    plt.xlabel('t[n]') 
    plt.ylabel('y[n]')     
    plt.title('Señal de salida para entrada senoidal - FIR')
    plt.legend()
    plt.show()
    
respuesta_2(sen_2, 10)

#respuesta al impulso, ya definí el impulso antes
B = np.zeros(N, dtype=float)
for n in range(N):
    if n in range(10):
        B[n] = P[n] 
    else:     
        B[n] = P[n] + 3 * P[n-10] 
        
plt.figure(11)        
plt.plot(tt, B, '*')
plt.grid(True)
plt.xlabel('t[n]') 
plt.ylabel('h[n]')      
plt.title('Respuesta al Impulso FIR')
plt.xlim([-0.001, 0.02])
plt.show()

def respuesta_3(señal, figure):
    F = np.zeros(N, dtype=float)
    for n in range(N):
        if n in range(10):
            F[n] = sen_2[n] 
        else:     
            F[n] = sen_2[n] + 3*F[n-10]          
    plt.figure(figure)        
    plt.plot(tt, F, 'x', label = 'Señal de salida')
    plt.plot(tt, sen_2, label = 'Señal de entrada') 
    plt.grid(True)
    plt.xlabel('t[n]') 
    plt.ylabel('y[n]')      
    plt.title('Señal de salida para entrada senoidal - IIR')
    plt.legend()
    plt.ylim([-2,2])
    plt.show()
    
respuesta_3(sen_2, 12)

#respuesta al impulso, ya definí impulso antes
G = np.zeros(N, dtype=float)
for n in range(N):
    if n in range(10):
        G[n] = P[n]
    else:     
        G[n] = P[n] + 3*G[n-10] 
        
plt.figure(13)        
plt.plot(tt, G, '*')
plt.grid(True)
plt.xlabel('t[n]') 
plt.ylabel('h[n]')  
plt.title('Respuesta al Impulso IIR')
plt.show()