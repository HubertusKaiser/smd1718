import numpy as np


def energy_mc(gamma, x):
    inv = (1-x)**(1/(1-gamma))
    return inv


def acceptance_mc(E):
    p = (1-np.exp(-E/2))**3
    return p


def hits_mc(E):
    return normal_polar(10*E, 10*E, 2*E, 2*E, 0)


def position_mc(mux, muy, sigmax, sigmay, rho):
    x, y = normal_polar(mux, muy, sigmax, sigmay, rho)
    # y, y_ = normal_polar(muy, sigma)
    for z in x:
        if z > 10:
            x, y = normal_polar(mux, muy, sigmax, sigmay, rho)
    for z in y:
        if z > 10:
            x, y = normal_polar(mux, muy, sigmax, sigmay, rho)
    return([x, y])


def normal_polar(mux, muy, sigmax, sigmay, rho):
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
    x_ms = np.sqrt(1-rho**2)*sigmax * x1 + rho*sigmay * x2 + mux
    y_ms = sigmay*x2 + muy
    return [x_ms, y_ms]

def normal_polar_std(N):
    x = np.empty(N)
    y = np.empty(N)
    for (u, v) in zip(x, y):
        u, v = get_random()
    return [x, y]
    
def get_random():
    s = 2
    while s > 1 or s < 0:
        u1 = np.random.uniform(0, 1, 1)
        u2 = np.random.uniform(0, 1, 1)
        v1 = 2*u1-1
        v2 = 2*u2-1
        s = v1**2+v2**2
        if np.isnan(np.sqrt(-2/s*np.log(s))):
            s = 2
    x1 = v1*np.sqrt(-2/s*np.log(s))
    x2 = v2*np.sqrt(-2/s*np.log(s))
    return [x1, x2]
    