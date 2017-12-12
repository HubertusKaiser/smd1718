import numpy as np
import uncertainties as unc

def aufg23():
    a_0 = unc.ufloat(1.0, 0.2)
    a_1 = unc.ufloat(1.0, 0.2)
    rho = -0.8
    def y(x):
        return a_0 + a_1*x

    
    
    
    
if __name__ == "__main__":
    aufg23()
    