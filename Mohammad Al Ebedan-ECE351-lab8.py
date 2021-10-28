# -*- coding: utf-8 -*-
"""
Created on Thu Oct  26 11:49:11 2021

@author: aleb9491
"""

################################################################
# Mohammad Al Ebedan                                           #
# ECE351-52                                                    #
# Lab 8                                                        #
# Oct 26, 2021                                                  #
################################################################

import numpy as np
import matplotlib.pyplot as plt 

#################################################################
#                           Part 1                              #
#################################################################
#%% part 1 task 1 

T = 8
w = 2*np.pi/T
steps = 1e-3
t = np.arange(0, 20, steps)


ak = []
bk = []

a0 = (1/T)*(T-T)
ak.append(a0)
for k in range(1, 2):
    a = 2/(k*w*T)*(2*np.sin(k*w*T/2) - np.sin(k*w*T))
    ak.append(a)

for k in range(1, 4):
    b = 2/(k*w*T)*(-2*np.cos(k*w*T/2) + np.cos(k*w*T) + 1)
    bk.append(b)

print("ak: ", ak)
print("bk: ", bk)

#%% part 1 task 2 

N = [1, 3, 15, 50, 150, 1500]

func = []
f = np.zeros(len(t))

for n in N:
    f = np.zeros(len(t))
    print(n)
    if n == 1 or n == 50:
        plt.figure(figsize=(10,7))
    if n == 1 or n == 50:
        plt.subplot(3, 1, 1)
        plt.title ('Fourier Series Approximation of a Square Wave')
        plt.plot(t,f)
    if n == 3 or n == 150:
        plt.subplot(3, 1, 2)
        
    if n == 15 or n == 1500:
        plt.subplot(3, 1, 3)
       
    for k in range(1, n+1):
        b = 2/(k*w*T)*(-2*np.cos(k*w*T/2) + np.cos(k*w*T) + 1)
        a = 2/(k*w*T)*(2*np.sin(k*w*T/2) - np.sin(k*w*T))
        f = f+b*np.sin(w*k*t)+a*np.cos(w*k*t)
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.plot(t, f)
    plt.grid()
plt.show()
