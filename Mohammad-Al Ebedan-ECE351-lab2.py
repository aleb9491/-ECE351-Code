# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
################################################################
# Mohammad Al Ebedan                                           #
# ECE351-02                                                    #
# Lab 2                                                        #
# Sep 9, 2021                                                  #
################################################################

import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sig
import control as con
import pandas as pd

#################################################################
#                           Part 1                              #
#################################################################
    
#the purpose is to be able to creat a simple user-defined function.


#%% task 2

steps = 1e-2

t = np.arange (0 , 5 + steps , steps )

def function1(t): 
    y = np.cos(t) 
    return y 

y = function1(t)

def u(t): 
    y = np.zeros((len(t),1))
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i]= 0
    return y        
   
steps = 1e-4
t = np.arange(0, 10 + steps, steps)
y = function1(t)



plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel ('y(t)')
plt.title ('cosine function')


#################################################################
#                           Part 2                              #
#################################################################



#%% task 2

def u(t): 
    y = np.zeros((len(t),1))
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i]= 0
    return y        
   
steps = 1e-4
t = np.arange(-1, 1 + steps, steps)
y = u(t)



plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel ('y(t)')
plt.title ('step')

def r(t): 
    y = np.zeros((len(t),1))
    for i in range(len(t)):
        if t[i]>=0:
            y[i] = t[i]
        else:
            y[i] = 0
    return y       

y = r(t)

plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel ('y(t)')
plt.title ('Ramp')
plt.show()


#%% task 3

def funct3(t):
    y = r(t)-r(t-3)+5*u(t-3)-2*u(t-6)-2*r(t-6)
    return y  

t = np.arange(-5, 10 + steps, steps)
y = funct3(t)

plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel ('y(t)')
plt.title ('Ramp')
plt.show()


#################################################################
#                           Part 3                              #
#################################################################

#%% task 1

t = np.arange(-10, 5 + steps, steps)

y = funct3(-t)

plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel ('y(t)')
plt.title ('time reversed Ramp')
plt.show()

#%% task 2

t = np.arange(-1, 14 + steps, steps)

y = funct3(t-4)

plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel ('y(t)')
plt.title ('time shifted Ramp')
plt.show()

t = np.arange(-14, 1 + steps, steps)

y = funct3(-t-4)

plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel ('y(t)')
plt.title ('time shifted reversed Ramp')
plt.show()

#%% task 3

t = np.arange(-5, 20 + steps, steps)

y = funct3(t/2)

plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel ('y(t)')
plt.title ('time scaled Ramp')
plt.show()

t = np.arange(-2, 5 + steps, steps)

y = funct3(2*t)

plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel ('y(t)')
plt.title ('time scaled Ramp')
plt.show()

#%% task 5


steps = 1e-3
t = np.arange(-5, 10+steps, steps)
y = funct3(t)
dt = np.diff(t)
dy = np.diff(y, axis = 0)/dt


plt.figure(figsize=(10,7))
plt.plot(t,y,'--', label = 'y(t)')
plt.plot(t[range(len(dy))], dy, label = 'dy(t)/dt')
plt.title ('Derivative WRT time')
#plt.legend()
plt.grid()
plt.ylim([-5,10])
plt.show()
