import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RNGs import phi_mc
from RNGs import acceptance_mc
from RNGs import hits_mc
from RNGs import position_mc
from RNGs import normal_polar

N = 10000
uni = np.random.uniform(0, 1, N)
E = phi_mc(2.7, 5, uni)  # Passt noch nicht
data = pd.DataFrame(E, columns=['Energy'])

acceptance_uni = np.random.uniform(0, 1, N)
diff = acceptance_uni - acceptance_mc(E)
acceptance_mask = [(x > 0) for x in diff]

data['AcceptanceMask'] = acceptance_mask

hits, hits_ = hits_mc(E)

data['NumberOfHits'] = hits

sigma_sig = 1/np.log10(hits+1)
x, y = position_mc(hits, 7, 3, sigma_sig)
data['x'] = x
data['y'] = y

plt.hist2d(x, y, bins=[30,30], range=[[6,8],[2,4]])
plt.colorbar()
plt.savefig('build/position_heatmap.pdf')
plt.clf()

plt.hist(x, bins=30, range=[0,10])
plt.savefig('build/text_x.pdf')
plt.clf()

plt.hist(y, bins=30, range=[0,10])
plt.savefig('build/text_y.pdf')
plt.clf()

mu_bkg = 2
sigma_bkg = 1
hits_bkg = []
for i in range(0, 10**3):
    p, p_ = normal_polar(mu_bkg, sigma_bkg)
    hits_bkg.append(10**np.array(p))  # hart ineffizient

hits_bkg = np.array(hits_bkg).flatten()
x_bkg, x_ = position_mc(hits_bkg, 5, 5, np.ones(10**3)*3)
y_bkg, y_ = position_mc(hits_bkg, 5, 5, np.ones(10**3)*3)
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