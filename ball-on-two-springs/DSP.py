import numpy as np
import cmath

def Hanning(t,t0,te):
    T = te-t0
    return 0.5*np.cos(2*np.pi/T*t+2*np.pi/T*(t0+te)/2)+0.5


def FFT(j,N,signal):
    sum = 0
    for k in range(N):
        c = 0 - 2j
        sum += signal[k] * (cmath.exp(c))**np.pi/N*j*k
    return sum.real
