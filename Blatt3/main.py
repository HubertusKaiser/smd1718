import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RNGs import energy_mc
from RNGs import acceptance_mc
from RNGs import hits_mc
from RNGs import position_mc
from RNGs import normal_polar


def ex_a():
    gamma_sig = 2.7
    uniform = np.random.uniform(0, 1, N_sig)
    E = energy_mc(gamma_sig, uniform)  # uses transformation method
    data['Energy'] = E


def ex_b():
    acceptance_uni = np.random.uniform(0, 1, N_sig)
    diff = acceptance_uni - acceptance_mc(data.Energy.values)  # rej method(?)
    acceptance_mask = [(x > 0) for x in diff]
    data['AcceptanceMask'] = acceptance_mask


def ex_c():
    hits, hits_ = hits_mc(data.Energy.values)  # uses polar method
    data['NumberOfHits'] = hits


def ex_d():
    mux_sig = 7  # mu of position normal dist
    muy_sig = 3
    sigma_sig = 1/np.log10(data.NumberOfHits.values+1)
    x, y = position_mc(mux_sig, muy_sig, sigma_sig, sigma_sig, 0)
    # y, y_ = position_mc(mux_sig, muy_sig, sigma_sig, 0)
    data['x'] = x
    data['y'] = y

    plt.hist2d(data.x.values, data.y.values, bins=[30, 30])
    plt.colorbar()
    plt.savefig('build/position_heatmap.pdf')
    plt.clf()


def ex_e():
    mu_bkgx = 2
    mu_bkgy = mu_bkgx
    N_bkg = 10**7
    sigma_bkgx = np.ones(N_bkg)
    sigma_bkgy = sigma_bkgx
    rho_bkghits = 0
    hits_bkg, hits_ = 10**np.array(normal_polar(mu_bkgx, mu_bkgy, sigma_bkgx, sigma_bkgy, rho_bkghits) ) # log10
    rho_bkgpos = 0.5
    x_bkg, y_bkg = position_mc(5, 5, np.ones(N_bkg)*3, np.ones(N_bkg)*3, rho_bkgpos)
    # y_bkg, y_ = position_mc(5, 5, np.ones(N_bkg)*3)
    bkg = pd.DataFrame()
    bkg['NumberOfHits'] = hits_bkg
    bkg['x'] = x_bkg
    bkg['y'] = y_bkg
    plt.hist2d(x_bkg, y_bkg, bins=[30, 30])
    plt.colorbar()
    plt.savefig('build/position_heatmap_bkg.pdf')
    plt.clf()

    plt.hist(np.log10(hits_bkg))
    plt.savefig('build/N_bkg.pdf')
    plt.clf()


if __name__ == "__main__":
    N_sig = 10000
    data = pd.DataFrame()
    ex_a()
    ex_b()
    ex_c()
    ex_d()
    ex_e()
    print(data)
