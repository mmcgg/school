
import numpy as np
import matplotlib.pyplot as plt

T = 5700
v = np.logspace(0,16,10000)
B = np.zeros(len(v))
c = 3e8
k = 1.38e-23
h = 6.626e-34

for i in xrange(0,len(v)-1):
    
    B[i] = 2*h*pow(v[i],3.0)/(pow(c,2)*np.exp((h*v[i])/(k*T))-1)

plt.loglog(v,B,basex=10,basey=10)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Brightness (W/m^2/Hz/sr)')
plt.title('Blackbody radiation curve for a star at 5700K')
plt.ion()
plt.show()

# ------------ Exponential Decay Plotting ----------------- #
T = 1.248e9
t = 4.5*np.linspace(0,1e9,10000)
N_K0 = 1

N_K = N_K0*pow(2,-t/T)
N_Ar = 1-N_K0*pow(2,-t/T)
ratio = N_Ar/N_K

plt.figure()
plt.plot(t,N_K)
plt.plot(t,N_Ar)
plt.legend(['K','Ar'])
plt.xlabel('Time (years)')
plt.ylabel('Abundance')
plt.title('Exponential decay of Potassium 40')
plt.show()

plt.figure()
plt.plot(t,ratio)
plt.legend(['Ar/K'])
plt.xlabel('Time (years)')
plt.ylabel('Ratio')
plt.title('Parent/Daughter Ratio')
plt.axhline(y=3)
plt.show()
plt.pause(10000.0)


