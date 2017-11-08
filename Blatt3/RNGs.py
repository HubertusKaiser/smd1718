import numpy as np


def phi(phi0, E, gamma):
    return (phi0*(E)**(-gamma))


def phi_mc(gamma, phi0, x):
    inv = (1-x)**(1-gamma)
    return inv


def acceptance_mc(E):
    p = (1-np.exp(-E/2))**3
    return p


def hits_mc(E):
    inside = False
    while inside is False:
        u1 = np.random.uniform(0, 1, 1)
        u2 = np.random.uniform(0, 1, 1)
        v1 = 2*u1-1
        v2 = 2*u2-1
        s = v1**2+v2**2
        if s < 1:
            inside = True
    x1 = v1*np.sqrt(-2/s*np.log(s))
    x2 = v2*np.sqrt(-2/s*np.log(s))  # Das sollten jetzt 2 normalverteilte ZZ sein?
    mu = 10*E
    sigma = 2*E
    x_E = sigma * x1 + sigma * x2 + mu
    return x_E