import numpy as np
import pandas as pd
from MCMC_template import MCMC
import matplotlib.pyplot as plt
import pylab as P


def aufg12():
    mu_0 = [0, 3]
    cov_0 = [[3.5, 0.9], [0.9, 2.6]]
    size = 10000
    P_0 = np.random.multivariate_normal(mu_0, cov_0, size)
    P_1_x = np.random.normal(6, 3.5, size)
    P_1_y = np.random.normal(-0.5 + 6*P_0[:,0], 1)
    P_1 = np.array([P_1_x, P_1_y])

    plt.figure(1)
    plt.scatter(P_1[0,:], P_1[1,:], s=0.5, c='g', alpha=0.8, label='P_0')
    plt.scatter(P_0[:,0], P_0[:,1], s=0.5, c='r', alpha=0.8, label='P_1')
    plt.xlabel('x-Achse')
    plt.ylabel('y-Achse')
    lgnd = plt.legend(loc='best')
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]
    plt.savefig('12b.pdf')
    plt.clf()

    data = pd.DataFrame()
    data['P_0_x'] = P_0[:,0]
    data['P_0_y'] = P_0[:,1]
    data['P_1_x'] = P_1[0,:]
    data['P_1_y'] = P_1[1,:]
    data.to_hdf('12_dataframe.hd5', 'data')
    return 1


def aufg13():
    def g1(x):
        return 0*x
        
    def g2(x):
        return -3/4*x

    def g3(x):
        return -5/4*x
            
            
    data = pd.read_hdf('12_dataframe.hd5')
    plt.figure(1)
    plt.scatter(data.P_0_x, data.P_0_y, s=0.5, c='g', alpha=0.8, label='P_0')
    plt.scatter(data.P_1_x, data.P_1_y, s=0.5, c='r', alpha=0.8, label='P_1')
    x = np.linspace(min(min(data.P_0_x), min(data.P_1_x)), max((max(data.P_0_x), max(data.P_1_x))), 1000)
    plt.plot(x, g1(x), label='g1', color = 'b')
    plt.plot(x, g2(x), label='g2', color = 'k')
    plt.plot(x, g3(x), label='g3', color = 'y')
    plt.xlabel('x-Achse')
    plt.ylabel('y-Achse')
    lgnd = plt.legend(loc='best')
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]
    plt.savefig('13_a.pdf')
    plt.clf()
    
    
    
    l1 = np.array([1, 0]) / (np.sqrt(1))
    P0_g1_x = (data.P_0_x*l1[0] + data.P_0_y*l1[1]) * l1[0]/(np.dot(l1, l1))
    P0_g1_y = (data.P_0_x*l1[0] + data.P_0_y*l1[1]) * l1[1]/(np.dot(l1, l1))
    P0_g1 = np.array(P0_g1_x, P0_g1_y)
    P1_g1_x = (data.P_1_x*l1[0] + data.P_1_y*l1[1]) * l1[0]/(np.dot(l1, l1))
    P1_g1_y = (data.P_1_x*l1[0] + data.P_1_y*l1[1]) * l1[1]/(np.dot(l1, l1))
    P1_g1 = np.array(P1_g1_x, P1_g1_y)
    plt.hist(P1_g1, bins = 40, histtype = 'step')
    plt.hist(P0_g1, bins = 40, histtype = 'step')
    plt.title('Projektion auf g1')
    plt.savefig('13_P0_g1.pdf')
    plt.clf()
       
    l2 = np.array([1, -3/4]) / (np.sqrt(1+9/16))
    P0_g2_x = (data.P_0_x*l2[0] + data.P_0_y*l2[1]) * l2[0]/(np.dot(l2, l2))
    P0_g2_y = (data.P_0_x*l2[0] + data.P_0_y*l2[1]) * l2[1]/(np.dot(l2, l2))
    P1_g2_x = (data.P_1_x*l2[0] + data.P_1_y*l2[1]) * l2[0]/(np.dot(l2, l2))
    P1_g2_y = (data.P_1_x*l2[0] + data.P_1_y*l2[1]) * l2[1]/(np.dot(l2, l2))
    P0_g2 = np.array(P0_g2_x, P0_g2_y)
    P1_g2 = np.array(P1_g2_x, P1_g2_y)
    plt.hist(P0_g2, bins = 40, histtype = 'step')
    plt.hist(P1_g2, bins = 40, histtype = 'step')
    plt.title('Projektion auf g2')
    plt.savefig('13_P0_g2.pdf')
    plt.clf()
    
    l3 = np.array([1, -5/4]) / (np.sqrt(1+25/16))
    P0_g3_x = (data.P_0_x*l2[0] + data.P_0_y*l3[0]) * l3[0]/(np.dot(l2, l2))
    P0_g3_y = (data.P_0_x*l2[0] + data.P_0_y*l3[1]) * l3[1]/(np.dot(l2, l2))
    P1_g3_x = (data.P_1_x*l2[0] + data.P_1_y*l3[0]) * l3[0]/(np.dot(l2, l2))
    P1_g3_y = (data.P_1_x*l2[0] + data.P_1_y*l3[1]) * l3[1]/(np.dot(l2, l2))
    P0_g3 = np.array(P0_g3_x, P0_g3_y)
    P1_g3 = np.array(P1_g3_x, P1_g3_y)
    plt.hist(P0_g3, bins = 40, histtype = 'step')
    plt.hist(P1_g3, bins = 40, histtype = 'step')
    plt.title('Projektion auf g3')
    plt.savefig('13_P0_g3.pdf')
    plt.clf()
    
   

def aufg15():
    size = 10000
    a = MCMC(step_size = 2, loc = -3, scale = 2)  
    y = a.sample(x0 = 15, n = size)
    
    n, bins, patches = plt.hist(y, bins = 30, normed=1)
    x = np.linspace(min(bins), max(bins), 1000)
    gauss = 1/np.sqrt(2*np.pi*(2)**2)*np.exp(-(x+3)**2/(2*2**2))
    
    plt.plot(x, gauss)
    plt.title('Normiertes Histogramm der MCMC Zufallszahlen')
    plt.xlabel('Generierte Zufallszahl')
    plt.ylabel('Normierte Wahrscheinlichkeit')
    plt.savefig('15_hist.pdf')
    plt.clf()
    
    x = np.arange(size)
    plt.plot(x, y, 'ko', markersize=0.5)
    plt.xlabel('Iterationsschritt')
    plt.ylabel('Zufallszahl im Schritt')
    plt.savefig('15_trace.pdf')
    plt.clf()
    
    plt.plot(x, y, 'ko', markersize=0.5)
    plt.xlabel('Iterationsschritt')
    plt.ylabel('Zufallszahl im Schritt')
    plt.xscale('log')
    plt.savefig('15_trace_logx.pdf')
    plt.clf()
    

if __name__ == "__main__":
    np.random.seed(123)
    aufg12()
    aufg13()
    aufg15()