import numpy as np
from uncertainties import unumpy
def aufg33():
    werte=np.array([31.6, 32.2, 31.2, 31.9, 31.3 , 30.8 ,31.3])
    err=np.ones(7)*0.5
    EW=31.3*np.ones(7)
    EW2=30.7*np.ones(7)
    werte=unumpy.uarray(werte, err)
#### Annahme dass die Werte normalverteilt sind --> Sigma=1
    a=np.square(werte-EW)
    b=np.square(werte-EW2)

    print("Aufgabe 33a,) das Chi^2")
    print(np.sum(a))
    print("Aufgabe 33b,) das Chi^2")
    print(np.sum(b))

if __name__ == '__main__':
    aufg33()
