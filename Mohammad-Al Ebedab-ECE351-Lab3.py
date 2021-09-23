# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:52:43 2021

@author: aleb9491
"""

################################################################
# Mohammad Al Ebedan                                           #
# ECE351-02                                                    #
# Lab 3                                                        #
# Sep 22, 2021                                                  #
################################################################

import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sig

#################################################################
#                           Part 1                              #
#################################################################

#%% Part 1 Task 1

def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

def r(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = t[i]
        else:
            y[i] = 0
    return y

def f_1(t):
    y = u(t - 2) - u(t - 9)
    return y

def f_2(t):
    y = np.exp(-t)*u(t)
    return y

def f_3(t):
    y = r(t - 2)*(u(t - 2) - u(t -3)) + r(4 - t)*(u(t - 3) - u(t - 4))
    return y

#%% Part 1 Task 2
    
steps = 1e-2
t = np.arange(0, 20 + steps, steps)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, f_1(t))
plt.ylabel('f_1(t)')
plt.title('Three User-Defined Functions')
plt.ylim([0, 1.2])
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(t, f_2(t))
plt.ylabel('f_2(t)')
plt.ylim([0, 1.2])
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(t, f_3(t))
plt.ylabel('f_3(t)')
plt.ylim([0, 1.2])
plt.grid()
plt.xlabel('t [s]')
plt.show()


#################################################################
#                           Part 2                              #
#################################################################

#%% Part 2 Task 1

def conv(f1, f2):
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1Extended = np.append(f1, np.zeros((1, Nf2 -1)))
    f2Extended = np.append(f2, np.zeros((1, Nf1 -1)))
    result = np.zeros(f1Extended.shape)
    
    for i in range(Nf2 + Nf1 - 2):
        result[i] = 0
        for j in range(Nf1):
            if(i - j + 1 > 0):
                try:
                    result[i] = result[i] + f1Extended[j]*f2Extended[i - j + 1]
                except:
                    print(i, j)
    return result

steps = 1e-2
t = np.arange(0, 20 + steps, steps)
tExtended = np.arange(0, 2*t[len(t) - 1], steps)

f1 = f_1(t)
f2 = f_2(t)
f3 = f_3(t)

#%% Part 2 Task 2

conv12 = conv(f1, f2)*steps
conv12Check = sig.convolve(f1, f2)*steps

plt.figure(figsize = (10, 7))
plt.plot(tExtended, conv12, label = 'User-Defined Convolution')
plt.plot(tExtended, conv12Check, '--', label = 'Built-In Convolution')
plt.ylim([0, 1.2])
plt.grid()
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('f_1(t) * f_2(t)')
plt.title('Convolution of f_1 and f_2')
plt.show()

#%% Part 2 Task 3

conv23 = conv(f2, f3)*steps
conv23Check = sig.convolve(f2, f3)*steps

plt.figure(figsize = (10, 7))
plt.plot(tExtended, conv23, label = 'User-Defined Convolution')
plt.plot(tExtended, conv23Check, '--', label = 'Built-In Convolution')
plt.ylim([0, 1.2])
plt.grid()
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('f_2(t) * f_3(t)')
plt.title('Convolution of f_2 and f_3')
plt.show()

#%% Part 2 Task 4

conv13 = conv(f1, f3)*steps
conv13Check = sig.convolve(f1, f3)*steps

plt.figure(figsize = (10, 7))
plt.plot(tExtended, conv13, label = 'User-Defined Convolution')
plt.plot(tExtended, conv13Check, '--', label = 'Built-In Convolution')
plt.ylim([0, 1.2])
plt.grid()
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('f_1(t) * f_3(t)')
plt.title('Convolution of f_1 and f_3')
plt.show()
