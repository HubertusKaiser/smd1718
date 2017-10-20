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

	print('Vergleich in 16bit Präzision:')
	plt.plot(x_16, f_direkt(x_16), 'kx', label='Direkt')
	plt.plot(x_16, f_naiv(x_16), 'rx', label='Naiv')
	plt.plot(x_16, f_horner(x_16), 'gx', label='Horner')
	plt.legend()
	plt.savefig('aufg1_16bit.pdf')
	plt.clf()
	
	print('Vergleich in 32bit Präzision:')
	plt.plot(x_32, f_direkt(x_32), 'kx', label='Direkt')
	plt.plot(x_32, f_naiv(x_32), 'rx', label='Naiv')
	plt.plot(x_32, f_horner(x_32), 'gx', label='Horner')
	plt.legend()
	plt.savefig('aufg1_32bit.pdf')
	plt.clf()
	
	print('Vergleich in 64bit Präzision:')
	plt.plot(x_64, f_direkt(x_64), 'kx', label='Direkt')
	plt.plot(x_64, f_naiv(x_64), 'rx', label='Naiv')
	plt.plot(x_64, f_horner(x_64), 'gx', label='Horner')
	plt.legend()
	plt.savefig('aufg1_64bit.pdf')
	plt.clf()
	

	
	
def aufg2():
	def f(x):
		y = (np.sqrt(9-x)-3)/x
		return y

	x = np.ones(20)
	for i in range(0,len(x)):
		x[i] = x[i]*10**(-i-1)
		
	plt.semilogx(x, f(x), 'ro', label='Numerische Werte')
	plt.axhline(-1/6, label='Grenzwert bei -1/6')
	plt.legend()
	plt.ylim(-0.5, 0.1)
	plt.savefig('aufg2.pdf')
	plt.clf()
	
	print(x)
	print("Der empirische Grenzwert ist -1/6")
	print("Der numerische Grenzwert i")
	
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
	#plt.ylim(0,0.15)
	plt.savefig('aufg3_f.pdf')
	plt.clf()
	
	plt.plot(x_g, np.absolute((g(x_g)-g_analytisch)/2), 'ko', label='Abweichung')
	plt.axhline(0.01)
	plt.legend(loc='best')
	plt.ylim(0,0.15)
	plt.savefig('aufg3_g.pdf')
	plt.clf()
	
	x_f_2 = np.linspace(-10**6, 10**6, 1000)
	plt.plot(x_f_2, f(x_f_2), 'ko', label='Funktionswert')
	#plt.axhline(0)
	plt.legend(loc='best')
	#plt.ylim(-0.15,0.15)
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
	
	
if __name__  == "__main__":

	aufg1()
	aufg2()
	aufg3()