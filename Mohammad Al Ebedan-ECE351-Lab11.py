# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 21:39:13 2021

@author: aleb9491
"""

################################################################
# Mohammad Al Ebedan                                           #
# ECE351-52                                                    #
# Lab 11                                                       #
# Nov 15, 2021                                                 #
################################################################

import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as sig
import zplane as zp

#%% task 3

num = [2, -40, 0]
num1 = [2, -40]
den = [1, -10, 16]

r, p, k = sig.residue(num1, den)
print("Residue: ", r)
print("Poles: ",p)
print("K: ",k)

#%% task 4

Z, P, K = zp.zplane(num, den)
                                 

#%% task 5

w, h = sig.freqz(num, den, whole=True) 
angles = np.unwrap(np.angle(h))

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w, h)
plt.xlabel("Frequency [rad/sample]")
plt.ylabel("Amplitude [dB]")
plt.grid()              
plt.subplot(2, 1, 2)              
plt.plot(w, angles)
plt.grid()
plt.xlabel("Frequency [rad/sample]")               
plt.ylabel('Angle (radians)')
plt.show()