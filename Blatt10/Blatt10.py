import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from math import factorial as fac


def aufg_31():
    x,y = np.genfromtxt('aufg_a.txt', unpack=True)
    x_lin = np.linspace(min(x), max(x), 100)
#    sns.set()
 #   sns.set_style('whitegrid')
    model = Pipeline([('poly', PolynomialFeatures(degree=6)),
                      ('linear', LinearRegression(fit_intercept=False))])
    
    model = model.fit(x.reshape(-1,1), y.reshape(-1,1))
    plt.plot(x,y, 'x')
    plt.plot(x_lin, model.predict(x_lin.reshape(-1,1)), 'r--')
    plt.savefig('linreg_sk.pdf')
    plt.clf()

    l = [0.1, 0.3, 0.7, 3, 10]
    
    def get_sq_error(x,x_pred):
        return np.mean((x.reshape(-1,1)-x_pred)**2)


    for a in l:
        model_ridge = Pipeline([('poly', PolynomialFeatures(degree=6)),
                            ('ridge', Ridge(alpha=a))])
        model_ridge = model_ridge.fit(x.reshape(-1,1), y.reshape(-1,1))
        print('mean squared error:', get_sq_error(y, model_ridge.predict(x.reshape(-1,1))))
        plt.plot(x_lin,model_ridge.predict(x_lin.reshape(-1,1)), '--', label='$\lambda$=' + str(a))
    
    plt.plot(x,y,'ko', label='Messwerte')
    plt.legend()
    plt.savefig('linreg_sk_ridge.pdf')
    plt.clf()



    data = pd.read_csv('aufg_c.csv')
    #data.drop('x')   ##wieso geht das nicht????
    y = data.mean(axis=1) # nur mit l46
    y_std = data.std(axis=1)
    
   # model = Pipeline([('poly', PolynomialFeatures(degree=6)),
    #                  ('ridge', Ridge(alpha = y_std))])   #sollen da dann die standardabweichungen hin?
    #model = model.fit(x.reshape(-1,1), y)
    #plt.plot(x, y, 'kx')
    #plt.plot(x, model.predict(x.reshape(-1,1)), 'r--')
    #plt.show()
    #plt.clf()


def aufg_29():
    x = np.linspace(1,25,100)
    c = np.log(fac(13)) + np.log(fac(8)) + np.log(fac(9))
    sns.set()   #damit funktioniert 'x' im plot nicht???
    plt.rc('text', usetex=True)
    def f1(x):
        res = 3*x-30*np.log(x)+c
        return res

    def f2(x):
        res = 3*x-30*(-(x-9)**2/2 + (x-9)) 
        return res

    def f3(x):
        res = 3*x - 30* (np.log(10) + 1/10*(x-10) -1/2*1/100*(x-10)**2) + c
        return res

    plt.plot(x,f1(x), 'k-', label='-lnL')
    plt.plot(x, f3(x), 'g-', label='taylor')
    a_05 = np.array([7.162, 13.504])
    a_2 = np.array([4.932, 17.723])
    a_92 = np.array([3.245, 22.696])
    plt.plot(a_05, f1(a_05), 'ro', label='a=0.5')
    plt.plot(a_2, f1(a_2), 'bo', label='a=2')
    plt.plot(a_92, f1(a_92), 'mo', label='a=9/2')
    plt.legend()
    plt.xlabel(r'$\lambda$')
    plt.ylabel('ln(L)')
    plt.ylim(-10,50)
    plt.axvline(10)
    plt.savefig('aufg29.pdf')

if __name__ == '__main__':
    aufg_31()
    aufg_29()
