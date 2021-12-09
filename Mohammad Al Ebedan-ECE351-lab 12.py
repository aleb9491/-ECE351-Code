# -*- coding: utf-8 -*-
"""
Created on Wed Nov  27 21:19:01 2021

@author: aleb9491
"""

################################################################
# Mohammad Al Ebedan                                           #
# ECE351-52                                                    #
# Lab 12                                                       #
# Nov 27, 2021                                                 #
################################################################

#################################################################
#                           Part 1                              #
#################################################################
#%% part 1

# the other packages you import will go here
import pandas as pd
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.fftpack 
import control as con 

# load input signal
df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values


plt.figure(figsize = (7, 5))
plt.plot(t, sensor_sig)
plt.grid()
plt.title('Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()

# your code starts here , good luck


fs = 1/(t[1]-t[0])

def fft(x, fs):
    N = len(x) 
    
    X_fft = scipy.fftpack.fft(x) 
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) 
    freq = np.arange(-N/2, N/2) * fs/N 

    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    for i in range(len(X_phi)):
        if np.abs(X_mag[i]) < 1e-10:
            X_phi[i] = 0 

    return freq, X_mag, X_phi
        
freq, X_mag, X_phi = fft(sensor_sig, fs)


def make_stem(ax, x, y, color = 'k', style = 'solid', label ='', linewidths = 2.5,** kwargs):
    ax.axhline(x[0], x[-1], 0, color = 'r')
    ax.vlines(x, 0, y, color = color, linestyles = style, label = label, linewidths = linewidths)
    ax.set_ylim([1.05*y.min(), 1.05*y.max()])


fig, ax = plt.subplots(figsize=(7, 3))
make_stem(ax, freq, X_mag)
plt.title("FFT of Input Signal")
plt.ylabel('Magnitude')
plt.xlabel('Freq(Hz)')
plt.show()

fig, ax = plt.subplots(figsize=(7, 3))
make_stem(ax, freq, X_mag)
plt.title("FFT of Low Frequency Noise")
plt.ylabel('Magnitude')
plt.xlabel('Freq(Hz)')
plt.xlim(0, 1800)
plt.show()

fig, ax = plt.subplots(figsize =(7, 3))
make_stem(ax, freq, X_mag)
plt.title("FFT of Position Signal")
plt.ylabel('Magnitude')
plt.xlabel('Freq(Hz)')
plt.xlim(1780, 2020)
plt.show()

fig, ax = plt.subplots(figsize =(7, 3))
make_stem(ax, freq, X_mag)
plt.title("FFT of High Frequency Noise")
plt.ylabel('Magnitude')
plt.xlabel('Freq(Hz)')
plt.xlim(45000, 55000)
plt.show()

#################################################################
#                           Part 2                              #
#################################################################
#%% part 2

steps = 1
w = np.arange(100, 6e5+steps, steps)
R = 6283
L = 1
C = 6.75e-9
num = [(1/2)*R/L, 0]
den = [1, R/L, 1/(L*C)]

#################################################################
#                           Part 3                              #
#################################################################
#%% part 3

plt.figure(figsize = (7, 4))

sys = con.TransferFunction(num, den)
_ = con.bode(sys, w, dB = True, Hz = True, deg = True, Plot = True)

plt.figure(figsize = (7, 4))
sys = con.TransferFunction(num, den)
_ = con.bode(sys, w, dB = True, Hz = True, deg = True, Plot = True)
plt.xlim(0, 1800)

plt.figure(figsize = (7, 4))
sys = con.TransferFunction(num, den)
_ = con.bode(sys, w, dB = True, Hz = True, deg = True, Plot = True)
plt.xlim(1700, 2100)

plt.figure(figsize = (7, 4))
sys = con.TransferFunction(num, den)
_ = con.bode(sys, w, dB = True, Hz = True, deg = True, Plot = True)
plt.xlim(45000, 55000)


#################################################################
#                           Part 4                              #
#################################################################
#%% part 4

znum, zden = sig.bilinear(num, den, fs=fs)
filtered_signal = sig.lfilter(znum, zden, sensor_sig)

freq, X_mag, X_phi = fft(filtered_signal, fs)

plt.figure(figsize = (7, 3))
plt.subplot(1, 1, 1)
plt.plot(t, filtered_signal)
plt.ylabel('y(t)')
plt.title('Filtered Input Signal')
plt.grid()
plt.xlabel('t(s)')

fig, ax = plt.subplots(figsize=(7, 3))
make_stem(ax, freq, X_mag)
plt.title("FFT of Filtered Input Signal")
plt.ylabel('Magnitude')
plt.xlabel('Freq(Hz)')
plt.show()

fig, ax = plt.subplots(figsize=(7, 3))
make_stem(ax, freq, X_mag)
plt.title("FFT of Filtered Low Frequency Noise")
plt.ylabel('Magnitude')
plt.xlabel('Freq(Hz)')
plt.xlim(0, 1800)
plt.show()

fig, ax = plt.subplots(figsize =(7, 3))
make_stem(ax, freq, X_mag)
plt.title("FFT of Filtered Position Signal")
plt.ylabel('Magnitude')
plt.xlabel('Freq(Hz)')
plt.xlim(1780, 2020)
plt.show()

fig, ax = plt.subplots(figsize =(7, 3))
make_stem(ax, freq, X_mag)
plt.title("FFT of Filtered High Frequency Noise")
plt.ylabel('Magnitude')
plt.xlabel('Freq(Hz)')
plt.xlim(45000, 55000)
plt.show()

filtered_dB=20*np.log10(X_mag);
plt.semilogx(freq,filtered_dB)
plt.ylabel('filtered magnitude in dB')
plt.xlabel('f [Hz]')
plt.grid()

print(np.max(filtered_dB))
