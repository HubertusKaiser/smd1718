import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from energy import phi_mc

uni = np.random.uniform(0, 1, 10**5)
E = phi_mc(2.7, 5, uni)  # Passt noch nicht

data = pd.DataFrame(E, columns = ['Energy'])
print(data)
print(data.mean())

plt.hist(E)
plt.show()
plt.clf()