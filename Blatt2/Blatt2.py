import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits .mplot3d import Axes3D


def linrandom(a,b,m,seed,N):
    x = [seed]
    for i in range(1, N):
        x.append(((a*x[-1]+b)%m)/m)
    return x

def scatter_2d(x,y,name):
    plt.scatter(x, y)
    plt.savefig(name+'.pdf')
    plt.clf()
    
    
def scatter_3d(x,y,z,name):
    fig = plt.figure ()
    ax = fig. add_subplot (111 , projection ='3d')
    ax.view_init(45, 30) # Elevation , Rotation
    ax.scatter(
        x, y, z,
        lw=0, # no lines around points
        s=5, # smaller points
                )
    plt.savefig(name +'.pdf')
    plt.clf()
    
def aufg9():

    a_primteilbar = [3,5,7,9,11,13]    # (a-1)/2 ganzzahlig
    random = []
    for a in a_primteilbar:
        random.append(linrandom(a,3,1024,0.5,1000))

    # Wie Periodenlände bestimmen?
    
    #Teilaufgabe c
    seeds=[0,1,0.5]
    for x0 in seeds:
        random = linrandom(1601, 3456, 10000, x0, 10000)
        plt.hist(random)
        plt.savefig('nr8_c_seed='+str(x0)+'.pdf')
        plt.clf()
    
    # Teilaufgabe d
    # 2D
    random = linrandom(1601, 3456, 10000, 0.2, 10000)
    random_0 = [random[x] for x in range(10000) if x%2==0]
    random_1 = [random[x] for x in range(10000) if x%2==1]
    scatter_2d(random_0, random_1, 'nr8_d_3D_seed='+str(0.2))

    
    #3D
    random_0 = [random[x] for x in range(10000) if x%3==0]
    random_1 = [random[x] for x in range(10000) if x%3==1]
    random_2 = [random[x] for x in range(10000) if x%3==2]
    random_0.pop()
    print(len(random_0))
    print(len(random_1))
    print(len(random_2))
    
    scatter_3d(random_0, random_1, random_2, 'nr8_d_3D_seed='+str(0.2))

        
aufg9()