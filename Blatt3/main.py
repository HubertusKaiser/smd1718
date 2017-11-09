import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RNGs import energy_mc
from RNGs import acceptance_mc
from RNGs import hits_mc
from RNGs import position_mc
from RNGs import normal_polar

N_sig = 10000
gamma_sig = 2.7
uniform = np.random.uniform(0, 1, N_sig)
E = energy_mc(gamma_sig, uniform)  # Passt noch nicht

plt.hist(E, bins=30, range=[0,30])
plt.savefig('build/E_test_sig.pdf')
plt.clf()

data = pd.DataFrame(E, columns=['Energy'])

acceptance_uni = np.random.uniform(0, 1, N_sig)
diff = acceptance_uni - acceptance_mc(data.Energy.values)
acceptance_mask = [(x > 0) for x in diff]

data['AcceptanceMask'] = acceptance_mask

hits, hits_ = hits_mc(data.Energy.values)

data['NumberOfHits'] = hits

mux_sig = 7
muy_sig = 3
sigma_sig = 1/np.log10(data.NumberOfHits.values+1)
x, y = position_mc(data.NumberOfHits.values, mux_sig, muy_sig, sigma_sig)
data['x'] = x
data['y'] = y

plt.hist2d(data.x.values, data.y.values, bins=[30,30], range=[[6,8],[2,4]])
plt.colorbar()
plt.savefig('build/position_heatmap.pdf')
plt.clf()

plt.hist(data.x.values, bins=30, range=[0,10])
plt.savefig('build/text_x.pdf')
plt.clf()

plt.hist(data.y.values, bins=30, range=[0,10])
plt.savefig('build/text_y.pdf')
plt.clf()

mu_bkg = 2
N_bkg = 10**3
sigma_bkg = np.ones(N_bkg)
hits_bkg, hits_ = normal_polar(mu_bkg, sigma_bkg)


hits_bkg = np.array(hits_bkg).flatten()
x_bkg, x_ = position_mc(hits_bkg, 5, 5, np.ones(N_bkg)*3)
y_bkg, y_ = position_mc(hits_bkg, 5, 5, np.ones(N_bkg)*3)
bkg = pd.DataFrame()
bkg['NumberOfHits'] = hits_bkg
bkg['x'] = x_bkg
bkg['y'] = y_bkg

plt.hist2d(x_bkg, y_bkg, bins=[30,30], range=[[6,8],[2,4]])
plt.colorbar()
plt.savefig('build/position_heatmap_bkg.pdf')
plt.clf()

plt.hist(x_bkg, bins=30, range=[0,10])
plt.savefig('build/text_x_bkg.pdf')
plt.clf()

plt.hist(y_bkg, bins=30, range=[0,10])
plt.savefig('build/text_y_bkg.pdf')
plt.clf()

plt.hist(np.log10(hits_bkg))
plt.savefig('build/N_bkg.pdf')
plt.clf()
