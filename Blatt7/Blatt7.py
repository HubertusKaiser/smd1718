import numpy as np
import uncertainties as unc
from uncertainties import unumpy as unp
import matplotlib.pyplot as plt 
from mc_class.py import MCMC


def aufg23():
    a_0 = unc.ufloat(1.0, 0.2)
    a_1 = unc.ufloat(1.0, 0.2)
    rho = -0.8
    def y(x):
        return a_0 + a_1*x

    x_test = np.linspace(-10, 10, 1000)
    y_err = unp.std_devs(y(x_test))
    plt.plot(x_test, y_err, 'kx')
    plt.savefig('23_a.pdf')
    
   # scs.norm(loc, scale).pdf
    a = MCMC(step_size = 2, loc = -3, scale = 2, y_pdf)  
    y = a.sample(x0 = 15, n = size)
    
if __name__ == "__main__":
    aufg23()
    