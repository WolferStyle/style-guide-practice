# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:31:51 2015

@author: Wolfer
"""

import numpy as np
import matplotlib.pyplot as plt
import solution_method


    
def RMS_calc(t,Phi):
    N = Phi.size
    Phi_A = solution_method.analyitical(t,Phi)
    rmsum = 0.0
    for i in range(0,N):
        rmsum += (Phi[i] - Phi_A[i])**2
    RMS = np.sqrt((1.0/N)*rmsum)
    return RMS




# ***************** BEGIN CALLING ABOVE FUNCTIONS ****************************

# Constants
L = 1.              # m
D = 0.005           # m^2/s
u = 0.2             # m/s
k = 2*np.pi/L       # m^-1
tau = 1/((k**2)*D)


C_lst = [0.1,   0.5,   2, 0.5, 0.5]
s_lst = [0.25, 0.25, .25, 0.5,   1]

case = 1
C = C_lst[case-1]
s = s_lst[case-1]
dx = C*D/(u*s)
dt = C*dx/u



#taustar = tau/d

t_T, Phi_T = solution_method.trapezoidal(C,s)
Phi_A_T = solution_method.analyitical(t_T,Phi_T)

t_C, Phi_C = solution_method.central_diff(C,s)
Phi_A_C = solution_method.analyitical(t_C,Phi_C)

t_U, Phi_U = solution_method.upwind(C,s)
Phi_A_U = solution_method.analyitical(t_U,Phi_U)

t_Q, Phi_Q = solution_method.quick(C,s)
Phi_A_Q = solution_method.analyitical(t_Q,Phi_Q)

x = np.linspace(0,1,Phi_A_T.size)

RMS = RMS_calc(t_T,Phi_T)
print '\nRMS for Trapazoidal = %e' %RMS
RMS = RMS_calc(t_C,Phi_C)
print '\nRMS for Central Differencing = %e' %RMS
RMS = RMS_calc(t_U,Phi_U)
print '\nRMS for Upwinding = %e' %RMS
RMS = RMS_calc(t_Q,Phi_Q)
print '\nRMS for QUICK = %e' %RMS


# Configure figures for production
WIDTH = 495.0  # the number latex spits out
FACTOR = 1.0   # the fraction of the width the figure should occupy
fig_width_pt = WIDTH * FACTOR
inches_per_pt = 1.0 / 72.27
golden_ratio = (np.sqrt(5) - 1.0) / 2.0      # because it looks good
fig_width_in = fig_width_pt * inches_per_pt  # figure width in inches
fig_height_in = fig_width_in * golden_ratio   # figure height in inches
fig_dims = [fig_width_in, fig_height_in]  # fig dims as a list

pfont = 11

lw = 1.5
ms=7

plt.figure(figsize=fig_dims)


# Constants
L = 1.              # m
D = 0.005           # m^2/s
u = 0.2             # m/s
k = 2*np.pi/L       # m^-1
tau = 1/((k**2)*D)


plt.plot(np.linspace(0,1,201),solution_method.analyitical(tau,np.linspace(0,1,201)),'k',linewidth=lw)
plt.plot(x,Phi_T,'s',linewidth=lw/2,markersize=ms,linestyle='--',dashes=(6,4))
plt.plot(x,Phi_C,'v',linewidth=lw/2,markersize=ms,linestyle='--',dashes=(6,4))
plt.plot(x,Phi_U,'^',linewidth=lw/2,markersize=ms,linestyle='--',dashes=(6,4))
plt.plot(x,Phi_Q,'o',linewidth=lw/2,markersize=ms,linestyle='--',dashes=(6,4))


plt.title('C = %.2f, s = %.2f' %(C,s),fontsize=pfont+1)
plt.xlabel(r'$x\hspace{0.3}[\mathrm{m}]$',fontsize=pfont+3,labelpad=0)
plt.ylabel(r'$\phi(x,t=\tau)$',fontsize=pfont+3,labelpad=0)
plt.tick_params(labelsize=pfont)
plt.legend(['Analytical','Trapezoidal','Central Diff.','Upwind','QUICK'],loc='lower left',fontsize=pfont+1)



plt.savefig('MethodCompareAlt_Case%d.eps' %case, format='eps',bbox_inches='tight')
