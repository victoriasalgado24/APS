#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 02:58:28 2025

@author: victoria24
"""

import wave as wv
import numpy as np
import matplotlib.pyplot as plt

corazon = wv.open("latidos_corazon.wav", 'rb')

frecuencia = corazon.getframerate()
N = corazon.getnframes()

canales = corazon.getnchannels()
print('cantidad de canales:', canales)

print('cantidad de frames:', N)
print('frecuencia de muestreo [Hz]:', frecuencia)

#una muestra por canal
bytes_por_muestra = corazon.getsampwidth()
print('bytes por muestra:', bytes_por_muestra)  

Ts = 1/frecuencia

vt = np.arange(0, N*Ts, Ts)

#quiero un array de tiempos y que para cada tiempo me diga que valor toma la señal que lei

frames_raw = corazon.readframes(corazon.getnframes())
signal = np.frombuffer(frames_raw, dtype=np.int16) # separa cada 16 bits(2 bytes)

canal_izq = signal[0::2]  
canal_der = signal[1::2]  

canal_izq_normalizado = canal_izq / 32768
canal_der_normalizado = canal_der / 32768

plt.figure(figsize=(13, 8))

# Canal izquierdo
plt.subplot(3, 1, 1)
plt.plot(vt, canal_izq_normalizado, color="blue")
plt.title("Canal izquierdo")
plt.xlabel("tiempo [s]")

# Canal derecho
plt.subplot(3, 1, 2)
plt.plot(vt, canal_der_normalizado, color="red")
plt.title("Canal derecho")
plt.xlabel("tiempo [s]")

# Ambos
plt.subplot(3, 1, 3)
plt.plot(vt, canal_izq_normalizado, label="Izquierdo", color="blue")
plt.plot(vt, canal_der_normalizado, label="Derecho", color="red", alpha=0.7)
plt.title("Ambos canales")
plt.xlabel("tiempo [s]")
plt.legend()

plt.tight_layout()
plt.show()

# Calcular energía (ejemplo con canal izquierdo)
energia = np.sum(canal_izq**2)
print('Energía total (canal izquierdo):', energia)

corazon.close()


