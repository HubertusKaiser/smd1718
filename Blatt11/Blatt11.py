import numpy as np


def chi_test(x,y,sigma):
    chi_squared = (y-x)**2/sigma**2
    return chi_squared


def aufg33():
    energy = np.array([31.6, 32.2, 31.2, 31.9, 31.3, 30.8, 31.3])
    err = np.ones(7)*0.5
    mean = np.mean(energy)
    std=np.std(energy)
    hyp_a = 31.3
    hyp_b = 30.7
    significance = 0.05
    chi_squared_a = chi_test(energy, hyp_a, std)   ##sollte die std nicht aus dem modell kommen?
    print(std)
    print(chi_squared_a)

if __name__=='__main__':
    aufg33()
