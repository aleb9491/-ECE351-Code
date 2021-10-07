# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 12:10:03 2021

@author: aleb9491
"""

###############################################
# Mohammad Al Ebedan                          #
# ECE351-52                                   #
# Lab 5                                       #
# Sep 30, 2021                                #
###############################################


import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sig

## The purpose of this lab 5 to be able to use Laplace transforms to find the 
# time-domain response of an RLC bandpass filter to impulse and step inputs.

#################################################################
#                           Part 1                              #
#################################################################

#%% part 1 task 1     
   
steps = 1e-5
t = np.arange(0, 1.2e-3 + steps, steps)

def h(t):
    h = (5000+1345j)*np.exp(-(5000+18584j)*t)+(5000-1345j)*np.exp((-5000-18584j)*t)
    return h
plt.figure(figsize=(10,7))
plt.subplot(1, 1, 1)
plt.plot(t,h(t))
plt.grid()
plt.ylabel ('h(t)')
plt.xlabel('t [s]')
plt.title ('Impulse Response by hand')


#%% part 1 task 2

R = 1000
L = 27e-3
c = 100e-9

num = [0 , 1/(R*c) , 0] 
den = [1 , 1/(c*R) , 1/(c*L)] 

tout , yout = sig.impulse(( num , den ) , T = t )

plt.figure(figsize=(10,7))
plt.subplot(1, 1, 1)
plt.plot(tout, yout)
plt.grid()
plt.ylabel ('h(t)')
plt.xlabel('t [s]')
plt.title ('Impulse Response')

#################################################################
#                           Part 2                              #
#################################################################
#%% part 2 task 1

num = [0 , 1/(R*c) , 0] 
den = [1 , 1/(c*R) , 1/(c*L)]

tout , yout = sig.step(( num , den ) , T = t )

plt.figure(figsize=(10,7))
plt.subplot(1, 1, 1)
plt.plot(tout, yout)
plt.grid()
plt.ylabel ('y(t)')
plt.xlabel('t [s]')
plt.title ('Step Response')
plt.show()

