import numpy as np


def energy_mc(gamma, x):
    inv = (1-x)**(1/(1-gamma))
    return inv


def acceptance_mc(E):
    p = (1-np.exp(-E/2))**3
    return p


def hits_mc(E):
    return normal_polar(10*E, 2*E)


def position_mc(N, mux, muy, sigma):
    # sigma = N
    x, x_ = normal_polar(mux, sigma)
    y, y_ = normal_polar(muy, sigma)
    for z in x:
        if z>10:
            z = normal_polar(7, sigma)
    for z in y:
        if z>10:
            z = normal_polar(3, sigma)
    return([x, y])


def normal_polar(mu, sigma):
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
    x2 = v2*np.sqrt(-2/s*np.log(s))
    x_ms = sigma * x1 + sigma * x2 + mu
    y_ms = sigma*x2 + mu
    return [x_ms, y_ms]
