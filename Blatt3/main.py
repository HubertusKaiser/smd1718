import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from energy import phi_mc
from energy import acceptance_mc
from energy import hits_mc

N = 100
uni = np.random.uniform(0, 1, N)
E = phi_mc(2.7, 5, uni)  # Passt noch nicht
data = pd.DataFrame(E, columns = ['Energy'])

acceptance_uni = np.random.uniform(0, 1, N)
diff = acceptance_uni - acceptance_mc(E)
acceptance_mask = [(x>0) for x in diff]

data['AcceptanceMask'] = acceptance_mask

hits = hits_mc(E)

data['NumberOfHits'] = hits

print(data)