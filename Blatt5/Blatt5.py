import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


def aufg16():
    data = pd.read_hdf('12_dataframe.hd5')
    # mu_0_x = np.mean(data.P_0_x)

    mu_0 = np.array([data['P_0_x'].mean(), data['P_0_y'].mean()])
    mu_1 = np.array([data['P_1_x'].mean(), data['P_1_y'].mean()])

    
    data_0 = data[['P_0_x', 'P_0_y']]
    data_1 = data[['P_1_x', 'P_1_y']]

    cov_0 = data_0.cov().values  # =S_0
    cov_1 = data_1.cov().values  # =S_1
    cov_01 = data.cov().values   # = S_B
    
    S_W = cov_0 + cov_1
    S_B = cov_01
    
    lambda_fisher = np.dot(np.linalg.inv(S_W), mu_0-mu_1)
    print(lambda_fisher)
    
    data_0_proj = np.dot(data_0.values, lambda_fisher)
    data_1_proj = np.dot(data_1.values, lambda_fisher)
    
    plt.hist(data_0_proj, histtype='step', bins=40)  #rot in B4
    plt.hist(data_1_proj, histtype='step', bins=40)  #grün in B4
    plt.savefig('fisherprojektion.pdf')
    plt.clf()
    
    
if __name__ == "__main__":
    np.random.seed(1234)
    aufg16()