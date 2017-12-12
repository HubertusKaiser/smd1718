import numpy as np
import uncertainties as unc
from uncertainties import unumpy as unp
import matplotlib.pyplot as plt 
from mc_class import MCMC
from scipy.stats import norm
import matplotlib.mlab as mlab



def aufg23():

    def y(x):
        return a_0 + a_1*x
    
    err = MCMC(step_size = 0.1, loc=1, scale=0.2)
    np.random.seed(123)
    err_1_vz = err.sample(x0 = 1, n=2000) -1
    err_1 = np.abs(err_1_vz)
    np.random.seed(231)
    err_2_vz = err.sample(x0 = 1, n=2000) -1
    err_2 = np.abs(err_2_vz)
    
    rho = 0.8
    plt.scatter(err_1_vz, err_2_vz, s=1)
    plt.savefig('23_scatter.pdf')
    plt.clf()
    
    x_lin = [-3, 0, 3]


    for x in x_lin:   
        y_mc = (err_1_vz+1) + (err_2_vz+1)*x
        n, bins, patches = plt.hist(y_mc, 50, normed=1)
        (mu, sigma) = norm.fit(y_mc)
        y = mlab.normpdf(bins, mu, sigma)
        lab = 'mu='+str(round(mu,3))+', sigma='+str(round(sigma,3))
        l = plt.plot(bins, y, 'r--', linewidth=2, label=lab)
        plt.legend()
        plt.savefig('23_c_'+str(x)+'.pdf')
        plt.clf()
    
    
if __name__ == "__main__":
    aufg23()
    