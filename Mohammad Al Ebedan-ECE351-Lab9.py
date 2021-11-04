# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 8:12:55 2021

@author: aleb9491
"""

################################################################
# Mohammad Al Ebedan                                           #
# ECE351-52                                                    #
# Lab 9                                                        #
# Nov 1, 2021                                                  #
################################################################

import numpy as np 
import matplotlib.pyplot as plt
import scipy.fftpack
from copy import deepcopy

#################################################################
#                           Part 1                              #
#################################################################
#%% task 1

fs = 100

def FFT(x, fs):
    N = len(x) # find the length of the signal
    X_fft = scipy.fftpack.fft(x) # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) # shift zero frequency components
    # to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N # compute the frequencies for the output
    # signal , (fs is the sampling frequency and
    # needs to be defined previously 
    X_mag = np.abs(X_fft_shifted)/N # compute the magnitudes of the signal
    X_phi = np.angle(X_fft_shifted) # compute the phases of the signal

    return X_mag, X_phi, freq
    # ----- End of user defined function ----- #

t = np.arange(0, 2, 1/fs)
y = np.cos(2*np.pi*t)
X_mag, X_phi, freq = FFT(y, fs)
plt.figure(figsize=(10,7))
plt.subplot("311")
plt.title("Cos(2*pi*t)")
plt.plot(t, y)
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.grid()
plt.subplot("323")
plt.stem(freq, X_mag, use_line_collection = True)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid()
plt.subplot("324")
plt.stem(freq, X_mag, use_line_collection = True)
plt.xlim([-2, 2])
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid()
plt.subplot("325")
plt.stem(freq, X_phi, use_line_collection = True)
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.grid()
plt.subplot("326")
plt.stem(freq, X_phi, use_line_collection = True)
plt.xlim([-2, 2])
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.grid()

#%% task 2

y = 5*np.sin(2*np.pi*t)
X_mag, X_phi, freq = FFT(y, fs)
plt.figure(figsize=(10, 7))
plt.subplot("311")
plt.title("5*Sin(2*pi*t)")
plt.plot(t, y)
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.grid()
plt.subplot("323")
plt.stem(freq, X_mag, use_line_collection = True)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid()
plt.subplot("324")
plt.stem(freq, X_mag, use_line_collection = True)
plt.xlim([-2, 2])
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid()
plt.subplot("325")
plt.stem(freq, X_phi, use_line_collection = True)
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.grid()
plt.subplot("326")
plt.stem(freq, X_phi, use_line_collection = True)
plt.xlim([-2, 2])
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.grid()

#%% task 3

y = 2*np.cos((2*np.pi*2*t) - 2) + (np.sin((2*np.pi*6*t)+3))**2
X_mag, X_phi, freq = FFT(y, fs)
plt.figure(figsize=(10, 7))
plt.subplot("311")
plt.title("2*Cos((2*pi*2*t) - 2) + (sin(2*pi*6*t) + 3))^2")
plt.plot(t, y)
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.grid()
plt.subplot("323")
plt.stem(freq, X_mag, use_line_collection = True)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid()
plt.subplot("324")
plt.stem(freq, X_mag, use_line_collection = True)
plt.xlim([-2, 2])
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid()
plt.subplot("325")
plt.stem(freq, X_phi, use_line_collection = True)
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.grid()
plt.subplot("326")
plt.stem(freq, X_phi, use_line_collection = True)
plt.xlim([-20, 2])
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.grid()

#%% task 4

fs = 100

def FFT(x, fs):
    N = len(x) # find the length of the signal
    X_fft = scipy.fftpack.fft(x) # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) # shift zero frequency components
    # to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N # compute the frequencies for the output
    # signal , (fs is the sampling frequency and
    # needs to be defined previously 
    X_mag = np.abs(X_fft_shifted)/N # compute the magnitudes of the signal
    X_phi = np.angle(X_fft_shifted) # compute the phases of the signal

    num = len(X_mag)
    X_phi_c = deepcopy(X_phi)

    for i in range(0, num):
        if X_mag[i]<1e-10:
            X_phi_c[i] = 0

    return X_mag, X_phi, X_phi_c, freq
    # ----- End of user defined function ----- #

t = np.arange(0, 2, 1/fs)
y = np.cos(2*np.pi*t)
X_mag, X_phi, X_phi_c, freq = FFT(y, fs)
plt.figure(figsize=(10,7))
plt.subplot("311")
plt.title("Cos(2*pi*t)")
plt.plot(t, y)
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.grid()
plt.subplot("323")
plt.stem(freq, X_mag, use_line_collection = True)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid()
plt.subplot("324")
plt.stem(freq, X_mag, use_line_collection = True)
plt.xlim([-2, 2])
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid()
plt.subplot("325")
plt.stem(freq, X_phi_c, use_line_collection = True)
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.grid()
plt.subplot("326")
plt.stem(freq, X_phi_c, use_line_collection = True)
plt.xlim([-2, 2])
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.grid()


y = 5*np.sin(2*np.pi*t)
X_mag, X_phi, X_phi_c, freq = FFT(y, fs)
plt.figure(figsize=(10, 7))
plt.subplot("311")
plt.title("5*Sin(2*pi*t)")
plt.plot(t, y)
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.grid()
plt.subplot("323")
plt.stem(freq, X_mag, use_line_collection = True)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid()
plt.subplot("324")
plt.stem(freq, X_mag, use_line_collection = True)
plt.xlim([-2, 2])
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid()
plt.subplot("325")
plt.stem(freq, X_phi_c, use_line_collection = True)
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.grid()
plt.subplot("326")
plt.stem(freq, X_phi_c, use_line_collection = True)
plt.xlim([-2, 2])
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.grid()


y = 2*np.cos((2*np.pi*2*t) - 2) + (np.sin((2*np.pi*6*t)+3))**2
X_mag, X_phi, X_phi_c, freq = FFT(y, fs)
plt.figure(figsize=(10, 7))
plt.subplot("311")
plt.title("2*Cos((2*pi*2*t) - 2) + (sin(2*pi*6*t) + 3))^2")
plt.plot(t, y)
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.grid()
plt.subplot("323")
plt.stem(freq, X_mag, use_line_collection = True)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid()
plt.subplot("324")
plt.stem(freq, X_mag, use_line_collection = True)
plt.xlim([-20, 20])
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid()
plt.subplot("325")
plt.stem(freq, X_phi_c, use_line_collection = True)
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.grid()
plt.subplot("326")
plt.stem(freq, X_phi_c, use_line_collection = True)
plt.xlim([-20, 20])
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.grid()

#%% task 5

T = 8
w = 2*np.pi/T
t = np.arange(0, 16, 1/fs)
N = 15

f = np.zeros(len(t))
for k in range(1, N+1):
    a = 2/(k*w*T)*(2*np.sin(k*w*T/2) - np.sin(k*w*T))
    b = 2/(k*w*T)*(-2*np.cos(k*w*T/2) + np.cos(k*w*T) + 1)
    f = f+b*np.sin(w*k*t)+a*np.cos(w*k*t)

X_mag, X_phi, X_phi_c, freq = FFT(f, fs)
plt.figure(figsize=(10, 7))
plt.subplot("311")
plt.title("Signal from Lab 8")
plt.plot(t, f)
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.grid()
plt.subplot("323")
plt.stem(freq, X_mag, use_line_collection = True)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid()
plt.subplot("324")
plt.stem(freq, X_mag, use_line_collection = True)
plt.xlim([-3, 3])
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid()
plt.subplot("325")
plt.stem(freq, X_phi_c, use_line_collection = True)
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.grid()
plt.subplot("326")
plt.stem(freq, X_phi_c, use_line_collection = True)
plt.xlim([-3, 3])
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.grid()
plt.show()
