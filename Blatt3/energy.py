def phi(phi0, E, gamma):
    return (phi0*(E)**(-gamma))


def phi_mc(gamma, phi0, x):
    inv = (1-x*gamma/phi0)**(gamma-1)
    A = phi0/(gamma+1)  # Normierung
    inv_norm = inv/A
    return inv_norm
