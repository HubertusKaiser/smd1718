import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits .mplot3d import Axes3D


def linrandom(a, b, m, seed, N, pl=False):
    x = [seed*m]
    periode = 0
    for i in range(1, N):
        n = ((a*x[-1]+b) % m)
        if pl is True and n in x:
            periode = i
            break
        else:
            x.append(n)
    r = np.array(x)/m
    return [r, periode]


def scatter_2d(x, y, name):
    plt.scatter(x, y)
    plt.savefig(name+'.pdf')
    plt.clf()


def scatter_3d(x, y, z, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, 30)  # Elevation , Rotation
    ax.scatter(
        x, y, z,
        lw=0,  # no lines around points
        s=5,  # smaller points
                )
    plt.savefig(name + '.pdf')
    plt.clf()


def aufg9():

    # Teilaufgabe a:
    a_test = np.arange(20)
    length = []
    for a in a_test:
        b, c = linrandom(a, 3, 1024, 0.5, 1100, pl=True)
        length.append(c)

    plt.plot(a_test, length, 'kx')
    plt.xlabel('Parameter a')
    plt.ylabel('Periodenlänge')
    plt.savefig('nr8_a.pdf')
    plt.clf()

    # Teilaufgabe c
    seeds = np.arange(10)/10
    for x0 in seeds:
        random, pl = linrandom(1601, 3456, 10000, x0, 10000, pl=False)
        plt.hist(random, bins=20)
        plt.savefig('nr8_c_seed='+str(x0)+'.pdf')
        plt.clf()

    # Teilaufgabe d
    # 2D
    random, pl = linrandom(1601, 3456, 10000, 0.2, 10000)
    random_numpy = np.random.uniform(0, 1, 10000)
    random_0 = [random[x] for x in range(10000) if x % 2 == 0]
    random_1 = [random[x] for x in range(10000) if x % 2 == 1]
    scatter_2d(random_0, random_1, 'nr8_d_2D_seed=' + str(0.2))

    # 3D
    random_0 = [random[x] for x in range(10000) if x % 3 == 0]
    random_1 = [random[x] for x in range(10000) if x % 3 == 1]
    random_2 = [random[x] for x in range(10000) if x % 3 == 2]
    random_0.pop()
    scatter_3d(random_0, random_1, random_2, 'nr8_d_3D_seed='+str(0.2))

    random_0 = [random_numpy[x] for x in range(10000) if x % 2 == 0]
    random_1 = [random_numpy[x] for x in range(10000) if x % 2 == 1]
    scatter_2d(random_0, random_1, 'nr8_d_npuni_2D_seed=' + str(0.2))

    # 3D
    random_0 = [random_numpy[x] for x in range(10000) if x % 3 == 0]
    random_1 = [random_numpy[x] for x in range(10000) if x % 3 == 1]
    random_2 = [random_numpy[x] for x in range(10000) if x % 3 == 2]
    random_0.pop()
    scatter_3d(random_0, random_1, random_2, 'nr8_d_npuni_3D_seed='+str(0.2))

    # Teilaufgabe f

    seeds = np.arange(0, 100)/100
    for x0 in seeds:
        random, pl = linrandom(1601, 3456, 10000, x0, 10000, pl=False)
        msk = [(y == 0.5) for y in random]
        find = [z for z in msk if z is True]
        plt.plot(x0, len(find), 'kx', markersize=5)
    plt.xlabel('Startwert')
    plt.ylabel('Anzahl der 1/2')
    plt.savefig('nr8_f.pdf')
    plt.clf()


aufg9()
