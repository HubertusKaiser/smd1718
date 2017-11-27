import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA


def aufg16():
    data = pd.read_hdf('12_dataframe.hd5')
    # mu_0_x = np.mean(data.P_0_x)

    mu_0 = np.array([data['P_0_x'].mean(), data['P_0_y'].mean()])
    mu_1 = np.array([data['P_1_x'].mean(), data['P_1_y'].mean()])
    print('a:')
    print(mu_0)
    print(mu_1)
    
    data_0 = data[['P_0_x', 'P_0_y']]
    data_1 = data[['P_1_x', 'P_1_y']]

    cov_0 = data_0.cov().values  # =S_0
    cov_1 = data_1.cov().values  # =S_1
    cov_01 = data.cov().values   # = S_B
    print('b:')
    print(cov_0)
    print(cov_1)
    
    
    S_W = cov_0 + cov_1
    S_B = cov_01
    print(S_W)
    print(S_B)
    
    
    lambda_fisher = np.dot(np.linalg.inv(S_W), (mu_0-mu_1))
    print(lambda_fisher)
    
    data_0_proj = np.dot(data_0.values, lambda_fisher)
    data_1_proj = np.dot(data_1.values, lambda_fisher)
   
    
    x = np.linspace(-10, 20, 1000)
    y = np.linspace(-20, 20, 1000)
    x_ = (x*lambda_fisher[0] + y*lambda_fisher[1]) / (lambda_fisher[0]**2 + lambda_fisher[1]**2)*lambda_fisher[0]
    y_ = (x*lambda_fisher[0] + y*lambda_fisher[1]) / (lambda_fisher[0]**2 + lambda_fisher[1]**2)*lambda_fisher[1]
    plt.scatter(data['P_0_x'], data['P_0_y'], s=0.5, c='g', alpha=0.8, label='P_0')
    plt.scatter(data['P_1_x'], data['P_1_y'], s=0.5, c='r', alpha=0.8, label='P_1')
    plt.plot(x_, y_, label='Fishergerade')  #???
    plt.xlabel('x-Achse')
    plt.ylabel('y-Achse')
    lgnd = plt.legend(loc='best')
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]
    plt.savefig('scatterplot.pdf')
    plt.clf()
   
    plt.hist(data_0_proj, histtype='step', bins=40)  #grün in B4?
    plt.hist(data_1_proj, histtype='step', bins=40)  #rot in B4?
    plt.savefig('fisherprojektion.pdf')
    plt.clf()


def aufg18():
    x,y = make_blobs(n_samples = 1000, centers = 2, n_features = 4, random_state=0)
    
    plt.scatter(x[:,0], x[:,1], s=0.5, c='g', alpha=0.8)
    plt.savefig('blobs.pdf')
    plt.clf()

    pca = PCA(n_components=4)
    transformed = pca.fit_transform(x)
    
    print(transformed.shape)
    plt.scatter(transformed[:,0], transformed[:,1], s=0.5, c='g', alpha=0.8)
    plt.savefig('blobs_transformed.pdf')
    plt.clf()
    
    eigenwerte = pca.components_.reshape((4, 4, 1))  #???
    print(eigenwerte)
    

   
if __name__ == "__main__":
    np.random.seed(1234)
    aufg16()
    aufg18()