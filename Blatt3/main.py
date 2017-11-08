import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from energy import phi_mc
from energy import acceptance_mc

uni = np.random.uniform(0, 1, 10**5)
E = phi_mc(2.7, 5, uni)  # Passt noch nicht

data = pd.DataFrame(E, columns = ['Energy'])
print(data.mean())

acceptance_uni = np.random.uniform(0, 1, 10**5)
diff = acceptance_uni - acceptance_mc(E)
acceptance_mask = [(x>0) for x in diff]

data['AcceptanceMask'] = acceptance_mask
print(data)