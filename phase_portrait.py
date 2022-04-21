import numpy
import scipy
import math
import matplotlib.pyplot as plt
from math import pi
from scipy.integrate import odeint
from mpl_toolkits.axisartist.axislines import SubplotZero
from pylab import rcParams

def System(v,t=0.):
    v1, v2 = v
    c = numpy.sqrt(3)
    p1 = v1/(1 + v1*v1)
    p2 = v2/(1 + v2*v2)
    m1 = abs(v1)
    m2 = abs(v2)
    a = c*v1*p2 + v2*p1 + v2*v2*p2
    ab = 0.5*(a + abs(a))
    
    if v2 > 0:
        r2 = 2
    else:
        r2 = 1
    
    b = c*m1 + r2*m2
    la = 1 - (ab/b)
    
    if b > 0:
        lg = numpy.log(la)
    else:
        lg = 0
    
    s1 = (lg/la) - c*m1
    s2 = (lg/la) - r2*m2
    e1 = numpy.exp(s1*(c*m1/b))
    e2 = numpy.exp(s2*(r2*m2/b))
    k1 = 1 - (1 - (ab/b)*(c*m1/b))*e1
    k2 = 1 - (1 - (ab/b)*(r2*m2/b))*e2
    q1 = -c*k1
    q2 = -r2*k2
    
    if v1 > 0:
        u1 = q1
    else:
        u1 = -q1
    
    if v2 > 0:
        u2 = q2
    else:
        u2 = -2*q2
    
    f1 = u2
    f2 =  0.5*c*u1 - 0.5*u2
    f3 = -0.5*c*u1 - 0.5*u2
    h1 = f1 - f2
    
    if h1 > 0:
        gh = f1
    else:
        gh = f2
    
    kh = gh - f3
    
    if kh > 0:
        mh = gh
    else:
        mh = f3
    
    if mh > 1:
        w1 = u1/mh
        w2 = u2/mh
    else:
        w1 = u1
        w2 = u2
    
    dv1 = 0.4*c*p2 + w1
    dv2 = 0.4*p1 + 0.4*v2*p2 + w2
    return (dv1, dv2)


n = 10
v1 = numpy.linspace(-4.0, 4.0, n)
v2 = numpy.linspace(-4.0, 4.0, n)
V1, V2 = numpy.meshgrid(v1, v2)
DV1, DV2 = numpy.zeros(V1.shape), numpy.zeros(V2.shape)

t = 0
for i in range(n):
    for j in range(n):
        v1 = V1[i, j]
        v2 = V2[i, j]
        DV1[i,j], DV2[i,j] = System([v1, v2], t)

plt.quiver(V1, V2, DV1, DV2, color='r')
plt.xlabel('$v_1$')
plt.ylabel('$v_2$')

for v0 in [[-3, -3], [-3, 0], [-3, 3], 
           [0, -3],  [0, 3], 
           [3, -3], [3, 0], [3, 3]]:
    t_param = numpy.linspace(0, 500, 10000)
    vt = odeint(System, v0, t_param)
    plt.plot(vt[:,0], vt[:,1], color = 'blue') # The curves.
    plt.plot([vt[0,0]], [vt[0,1]], 'o', color='black') # Inicial condition.

plt.savefig('./phase-portrait.png', dpi = 600)
