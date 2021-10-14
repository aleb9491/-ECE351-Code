# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:49:11 2021

@author: aleb9491
"""

################################################################
# Mohammad Al Ebedan                                           #
# ECE351-52                                                    #
# Lab 6                                                        #
# Oct 7, 2021                                                  #
################################################################

import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sig

#################################################################
#                           Part 1                              #
#################################################################


#%% part 1 task 1 

def u(t): 
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i]= 0
    return y        
   
steps = 1e-5
t = np.arange(0, 2 + steps, steps)

def h(t):
    h = (1/2 - (1/2)*np.exp(-4*t)+np.exp(-6*t))*u(t)
    return h
plt.figure(figsize=(10,7))
plt.subplot(1, 1, 1)
plt.plot(t,h(t))
plt.grid()
plt.ylabel ('y(t)')
plt.xlabel('t [s]')
plt.title ('Step Response by hand')

#%% part 1 task 2 

num = [1 , 6 , 12] 
den = [1 , 10 , 24]

tout , yout = sig.step(( num , den ) , T = t )

plt.figure(figsize=(10,7))
plt.subplot(1, 1, 1)
plt.plot(tout, yout)
plt.grid()
plt.ylabel ('y(t)')
plt.xlabel('t [s]')
plt.title ('Step Response')


#%% part 1 task 3 

num = [0, 1 , 6 , 12] 
den = [1 , 10 , 24, 0]
    
R, P, K = sig.residue(num, den)   
      
print("Residues:\n", R)

print("Poles:\n", P)

print("Gain:\n", K)


#################################################################
#                           Part 2                              #
#################################################################
#%% part 2 task 1

num = [25250] 
den = [1 , 18 , 218, 2036, 9085, 25250, 0]
    
R1, P1, K1 = sig.residue(num, den)   
      
print("Residues:\n", R1)

print("Poles:\n", P1)

print("Gain:\n", K1)

#%% part 2 task 2

def cosine_method(R1,P1,t):
    y = 0 
    for i in range(len(R1)):
        kmag = np.abs(R1[i])
        kang = np.angle(R1[i])
        alpha = np.real(P1[i])
        omega = np.imag(P1[i])
        y = y + kmag*np.exp(alpha*t)*np.cos(omega*t + kang)*u(t) 
    return y
   
t =np.arange(0, 4.5 + steps, steps)
plt.figure(figsize=(10,7))
plt.subplot(1,1,1)
plt.plot(t,cosine_method(R1,P1,t))
plt.grid()
plt.ylabel ('y(t)')
plt.xlabel('t [s]')
plt.title ('Step Response')
plt.show()

#%% part 2 task 3

num = [25250] 
den = [1 , 18 , 218, 2036, 9085, 25250]

tout , yout = sig.step((num , den) , T = t)

plt.figure(figsize=(10,7))
plt.subplot(1, 1, 1)
plt.plot(tout, yout)
plt.grid()
plt.ylabel ('y(t)')
plt.xlabel('t [s]')
plt.title ('Step Response H(s)')
plt.show()