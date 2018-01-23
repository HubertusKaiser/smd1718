import numpy as np
from math import factorial as fac


def aufg32():
    def log_lik(N_on, N_off, a, b, s):
        f = N_off*np.log(b)+N_on*np.log(s+a*b)-(1+a)*b-s-np.log(fac(N_off))-np.log(fac(N_on))
        return f

    def s_max(N_on, N_off, a):
        return N_on-a*N_off

    def b_max(N_off):
        return N_off

    ###nullhypothese s_0=0

if __name__=='__main__':
    aufg32()
