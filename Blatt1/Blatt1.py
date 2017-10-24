import numpy as np
import matplotlib.pyplot as plt


def aufg1():
    def f_direkt(x):
        y = (1-x)**6
        return (y)

    def f_naiv(x):
        y = 1-6*x+15*x**2-20*x**3+15*x**4-6*x**5+x**6
        return(y)

    def f_horner(x):
        y = 1+x*(-6+x*(15+x*(-20+x*(15+x*(-6+x)))))
        return(y)

    x_16 = np.linspace(0.999, 1.001, 1000, dtype='float16')
    x_32 = np.linspace(0.999, 1.001, 1000, dtype='float32')
    x_64 = np.linspace(0.999, 1.001, 1000, dtype='float64')

    plt.plot(x_16, f_direkt(x_16), 'ko', label='Direkt')
    plt.plot(x_16, f_naiv(x_16), 'ro', label='Naiv')
    plt.plot(x_16, f_horner(x_16), 'go', label='Horner')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    # plt.xlim(x_16[0]*0.9, x_16[999]*1.1)
    plt.savefig('aufg1_16bit.pdf')
    plt.clf()

    plt.plot(x_32, f_direkt(x_32), 'kx', label='Direkt')
    plt.plot(x_32, f_naiv(x_32), 'rx', label='Naiv')
    plt.plot(x_32, f_horner(x_32), 'gx', label='Horner')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    # plt.xlim(x_32[0]*0.9, x_32[999]*1.1)
    plt.savefig('aufg1_32bit.pdf')
    plt.clf()

    plt.plot(x_64, f_direkt(x_64), 'kx', label='Direkt')
    plt.plot(x_64, f_naiv(x_64), 'rx', label='Naiv')
    plt.plot(x_64, f_horner(x_64), 'gx', label='Horner')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    # plt.xlim(x_64[0]*0.9, x_64[999]*1.1)
    plt.savefig('aufg1_64bit.pdf')
    plt.clf()


def aufg2():
    def f(x):
        y = (np.sqrt(9-x)-3)/x
        return y

    x = np.ones(20)
    for i in range(0, len(x)):
        x[i] = x[i]*10**(-i-1)

    plt.semilogx(x, f(x), 'ro', label='Numerische Werte')
    plt.axhline(-1/6, label='Grenzwert für x->0')
    plt.legend()
    plt.ylim(-0.5, 0.1)
    plt.xlim(x[0]*0.9, x[19]*1.1)
    plt.savefig('aufg2.pdf')
    plt.clf()


def aufg3():
    def f(x):
        y = (x**3+1/3)-(x**3-1/3)
        return y

    def g(x):
        y = ((3+x**3)-(3-x**3))/x**3
        return(y)

    f_analytisch = 2/3
    g_analytisch = 2
    x = np.linspace(-5*10**4, 5*10**4, 1000)
    x_g = np.linspace(-0.00005, 0.00005, 1000)

    plt.plot(x, np.absolute((f(x)-f_analytisch)/2*3), 'ko', label='Abweichung')
    plt.axhline(0.01)
    plt.legend(loc='best')
    # plt.ylim(0,0.15)
    plt.savefig('aufg3_f_Abweichung.pdf')
    plt.clf()

    plt.plot(x_g, np.absolute((g(x_g)-g_analytisch)/2), 'ko',
             label='Abweichung')
    plt.axhline(0.01)
    plt.legend(loc='best')
    plt.ylim(0, 0.15)
    plt.savefig('aufg3_g_Abweichung.pdf')
    plt.clf()

    x_f_2 = np.linspace(-10**6, 10**6, 1000)
    plt.plot(x_f_2, f(x_f_2), 'ko', label='Funktionswert')
    plt.axhline(0)
    plt.legend(loc='best')
    plt.ylim(-0.05, 1.05)
    plt.savefig('aufg3_f_0.pdf')
    plt.clf()

    x_g_2 = np.linspace(-0.00002, 0.00002, 1000)
    plt.plot(x_g_2, g(x_g_2), 'ko', label='Funktionswert')
    plt.axhline(0)
    plt.legend(loc='best')
    plt.ylim(-0.1, 4)
    plt.xlim(-0.00002, 0.00002)
    plt.savefig('aufg3_g_0.pdf')
    plt.clf()

def aufg4():
	#Definition der Größen der Funktion
	#wq = Wirkungsquerschnitts
	#fwq = Funktion des Wirkungsquerschnitts
	#alpha= Feinstrukturkonstante
	alpha= 1/137
	#Ee = Energie des Elektron [Ee]=MeV
	Ee=50
	#Masse des Elektron [me]=MeV
	me=0.000511
	#ß beta
	ß=np.sqrt(1-me/Ee)

	def fwq(th):
		wq = ((alpha)**2 / Ee) * (2+ (np.sin(th)**2))/(1-ß**2 *np.cos(th)**2)
		return wq
	th=np.linspace(0,np.pi, 100000)
	plt.plot(th, fwq(th))
    plt.ylim(0,0.01)
    plt.savefig("aufg4(schlecht_konditioniert).pdf")



if __name__ == "__main__":
    aufg1()
    aufg2()
    aufg3()
    aufg4()
