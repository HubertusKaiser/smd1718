import numpy as np
import uncertainties as unc
from uncertainties import unumpy as unp
import matplotlib.pyplot as plt 
from mc_class import MCMC
from scipy.stats import norm
import matplotlib.mlab as mlab



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
    plt.clf()
   # scs.norm(loc, scale).pdf
    #a = MCMC(step_size = 2, loc = -3, scale = 2, y)  
    #y = a.sample(x0 = 15, n = size)
    
    err = MCMC(step_size = 0.1, loc=0, scale=0.2)
    np.random.seed(123)
    err_1_vz = err.sample(x0 = 0, n=10000)
    err_1 = np.abs(err_1_vz)
    np.random.seed(231)
    err_2_vz = err.sample(x0 = 0, n=10000)
    err_2 = np.abs(err_2_vz)
    
    x = -1
    rho = 0.8
    y_err_mc = np.sqrt(err_1**2 + err_2**2*x**2+ 2*x*rho*err_1*err_2)  
    #print(y_err_mc)
    #print(y_err_mc)
    
    #plt.hist(y_err_mc)
    #plt.show()

    #print(mu, sigma)

    plt.scatter(err_1_vz, err_2_vz, s=1)
    plt.savefig('23_scatter.pdf')
    plt.clf()
    
    x_lin = np.linspace(-3,3,7)


    for x in x_lin:   
        #y_err_mc = np.sqrt(err_1_vz**2 + err_2_vz**2*x**2+ 2*x*rho*err_1_vz*err_2_vz)  
        y_err_mc = np.sqrt(err_1**2 + err_2**2*x**2+ 2*x*rho*err_1*err_2)  
        y_err_mc = np.concatenate((y_err_mc, -y_err_mc))   
        n, bins, patches = plt.hist(y_err_mc, 60, normed=1, facecolor='green', alpha=0.75)
        (mu, sigma) = norm.fit(y_err_mc)
        y = mlab.normpdf(bins, mu, sigma)
        lab = 'mu='+str(round(mu,3))+', sigma='+str(round(sigma,3))
        l = plt.plot(bins, y, 'r--', linewidth=2, label=lab)
        plt.legend()
        plt.title('Fehler von y für x=' + str(x))
        plt.savefig('hist_normed_' + str(x) + '.pdf')
        plt.clf()
    
if __name__ == "__main__":
    aufg23()
    