def phi(phi0, E, gamma):
    return (phi0*(E)**(-gamma))


def phi_mc(gamma, phi0, x):
    inv = (1-x)**(1-gamma)
    return inv
