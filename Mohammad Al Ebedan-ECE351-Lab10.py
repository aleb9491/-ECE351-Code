# -*- coding: utf-8 -*-
"""
Created on Tue Nov  10 9:20:13 2021

@author: aleb9491
"""

################################################################
# Mohammad Al Ebedan                                           #
# ECE351-52                                                    #
# Lab 10                                                       #
# Nov 9, 2021                                                  #
################################################################

import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con

#################################################################
#                           Part 1                              #
#################################################################
#%% part 1 task 1

R = 1000
L = 27e-3
C = 100e-9

w = np.arange(10e3, 10e6, 5)

mag = (w*L)/(np.sqrt(((R-R*L*C*w**2)**2)+(L*w)**2))
phase = 90-np.arctan((w*L)/(R-R*L*C*w**2))

plt.figure()
plt.subplot(2, 1, 1)
plt.semilogx(w, mag)
plt.xlabel('rad/s')
plt.ylabel('magnitude')
plt.grid()
plt.subplot(2, 1, 2)
plt.semilogx(w, phase)
plt.xlabel('rad/s')
plt.ylabel('phase')
plt.grid()

#%% part 1 task 2

num = [1/(R*C)]
den = [1, 1/(R*C), 1/(L*C)]

w, mag, phase = sig.bode((num, den), w = w)

plt.figure()
plt.subplot(2, 1, 1)
plt.semilogx(w, mag)
plt.xlabel('rad/s')
plt.ylabel('magnitude')
plt.grid()
plt.subplot(2, 1, 2)
plt.semilogx(w, phase)
plt.xlabel('rad/s')
plt.ylabel('phase')
plt.grid()

#%% part 1 task 3

plt.figure()
sys = con.TransferFunction ( num , den )
mag, phase, omega = con.bode(sys , w , dB = True , Hz = True , deg = True , Plot = True )


#################################################################
#                           Part 2                              #
#################################################################
#%% part 2 task 1

fs = 1000000
T = 1/fs
t = np.arange(0, 0.01, T)
x = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)

plt.figure()
plt.plot(t, x)
plt.xlabel('Time')
plt.ylabel('magnitude')
plt.grid()

#%% part 2 task 2,3 and 4

z, p = sig.bilinear(num, den, fs = fs)
y = sig.lfilter(z, p, x)

plt.figure()
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('magnitude')
plt.grid()
plt.show()
