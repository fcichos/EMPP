# %% Cell 1
#

import matplotlib.pyplot as plt
import numpy as np
# %% Cell 2
#
# numerische Differentiation
#


def f(x,a,b,c):
    return a*x**3-b*x**2+c*x-1


x=np.linspace(-2*np.pi,2*np.pi,1000)
xs=np.linspace(-2*np.pi,2*np.pi,1000)
plt.plot(x,f(x,0.1,1,1))
plt.show()

# %% Cell 3
#

def integate(f,x,*params):
    print(*params)
    return np.sum(f(x,*params)[:-1]*(x[1:]-x[:-1]))
# %% Cell 4
#

integate(f,x,0.1,1,1)
