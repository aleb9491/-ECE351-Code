# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:46:13 2021

@author: aleb9491
"""

################################################################
# Mohammad Al Ebedan                                           #
# ECE351-52                                                    #
# Lab 7                                                        #
# Oct 18, 2021                                                 #
################################################################

import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

################################################################
#                           Part 1                             #
################################################################
#%% task 2

numG = [1, 9]
denG = [1 , -2 , -40, -64]
z1, p1, k1 = sig.tf2zpk(numG, denG)
print("\nFor G(s):")
print("Zeros:\n", z1)
print("Poles:\n", p1)
print("Gain:\n", k1)

numA = [1, 4]
denA = [1 , 4 , 3]
z2, p2, k2 = sig.tf2zpk(numA, denA)
print("\nFor A(s):")
print("Zeros:\n", z2)
print("Poles:\n", p2)
print("Gain:\n", k2)

B = [1, 26, 168]
rootsB = np.roots(B)
print("\nFor B(s):")
print("Roots:\n", rootsB)


#%% task 5

#OLTF: open-loop transfer function
#CLTF: closed-loop transfer function

OLTFNum = sig.convolve(numG, numA)
OLTFDen = sig.convolve(denG, denA)

tout, yout = sig.step((OLTFNum, OLTFDen))

plt.plot(tout, yout)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Step respone(open loop)")

################################################################
#                           Part 2                             #
################################################################
#%% task 2

CLTFNum = sig.convolve(numG, numA)
print("\nNumerator")
print(CLTFNum)

CLTFDen = sig.convolve(denG + sig.convolve(numG, B), denA)
print("\nDenominator")
print(CLTFDen)

z3, p3, k3 = sig.tf2zpk(CLTFNum, CLTFDen)
print("\nFor Close loop transfer function:\n")
print("Zeros:\n", z3)
print("Poles:\n", p3)
print("Gain:\n", k3)

#%% task 4

tout, yout = sig.step((CLTFNum, CLTFDen))

plt.figure()
plt.plot(tout, yout)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Step respone(closed loop)")
plt.show()






