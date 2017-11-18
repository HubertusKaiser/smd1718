import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RNGs import energy_mc
from RNGs import acceptance_mc
from RNGs import hits_mc
from RNGs import position_mc
from RNGs import normal_polar
from RNGs import normal_polar_std

def ex_a():
    gamma_sig = 2.7
    uniform = np.random.uniform(0, 1, N_sig)
    E = energy_mc(gamma_sig, uniform)  # uses transformation method
    data['Energy'] = E
    plt.hist(E, bins = 20)
    plt.savefig('build/E.pdf')
    plt.clf()


def ex_b():
    acceptance_uni = np.random.uniform(0, 1, N_sig)
    diff = acceptance_uni - acceptance_mc(data.Energy.values)  # rej method(?)
    acceptance_mask = [(x > 0) for x in diff]
    data['AcceptanceMask'] = acceptance_mask
    # wie plotten?


def ex_c():
    hits, hits_ = hits_mc(data.Energy.values)  # uses polar method
    data['NumberOfHits'] = np.round(hits)


def ex_d():
    mux_sig = 7 
    muy_sig = 3
    sigma_sig = 1/np.log10(data.NumberOfHits.values+1)
    x, y = normal_polar_std(N_sig) #MCMC?
    x = x*sigma_sig + mux_sig
    y = y*sigma_sig + muy_sig
    for (a,b) in zip(x,y):   # hart ineffizient
        inside = False
        while not inside:
            if a.all() < 10 and b.all() < 10 and a.all() > 0 and b.all() > 0:
                inside = True
            if a.all() > 10 or a.all() < 0:
                a = normal_polar_std(1) * sigma_sig + mux_sig
            if b.all() > 10 or b.all() < 0:
                b = normal_polar_std(1) * sigma_sig + muy_sig
    data['x'] = x
    data['y'] = y

    plt.hist2d(data.x.values, data.y.values, bins=[30, 30])
    plt.colorbar()
    plt.savefig('build/position_heatmap.pdf')
    plt.clf()


def ex_e():
    mu_bkgx = 2
    mu_bkgy = mu_bkgx
    N_bkg = 10**3
    sigma_bkgx = np.ones(N_bkg)
    sigma_bkgy = sigma_bkgx
    rho_bkghits = 0
    t,t2 = normal_polar_std(N_bkg)
    hits_bkg_log = t*2+1 # log10
    rho = 0.5
    x_bkg, y_bkg = normal_polar_std(N_bkg)
    x_bkg = x_bkg * sigma_bkgx * np.sqrt(1-rho**2) + rho * sigma_bkgy * y_bkg + mu_bkgx
    y_bkg = y_bkg * sigma_bkgy + mu_bkgy
    bkg['NumberOfHits_log'] = hits_bkg_log
    bkg['x'] = x_bkg
    bkg['y'] = y_bkg
    #plt.hist2d(x_bkg, y_bkg, bins=[30, 30])
    #plt.colorbar()
    #plt.savefig('build/position_heatmap_bkg.pdf')
    #plt.clf()

    #plt.hist(t, bins = 20)
    #plt.savefig('build/N_bkg.pdf')
    #plt.clf()


if __name__ == "__main__":
    np.random.seed(12)
    N_sig = 10000
    data = pd.DataFrame()
    bkg = pd.DataFrame()
    ex_a()
    ex_b()
    ex_c()
    ex_d()
    ex_e()
    data.to_hdf('build/NeutrinoMC.hd5', 'data')
    bkg.to_hdf('build/NeutrinoMC.hd5', 'bkg')
