# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:33:59 2015

@author: Wolfer
"""

import numpy as np
import matplotlib.pyplot as plt
import solution_method
from scipy import stats


def rms_calc(t,Phi):
    """Returns RMS error between numerical and analytical solution."""
    N = Phi.size
    Phi_A = solution_method.analyitical(t,Phi)
    rmsum = 0.0
    for i in range(0,N):
        rmsum += (Phi[i] - Phi_A[i])**2
    RMS = np.sqrt((1.0/N)*rmsum)
    return RMS


# Constants
L = 1.              # m
D = 0.005           # m^2/s
u = 0.2             # m/s
k = 2*np.pi/L       # m^-1
tau = 1/((k**2)*D)

# Configure figures for production
WIDTH = 495.0  # the number latex spits out
FACTOR = 1.0   # the fraction of the width the figure should occupy
fig_width_pt = WIDTH * FACTOR
inches_per_pt = 1.0 / 72.27
golden_ratio = (np.sqrt(5) - 1.0) / 2.0      # because it looks good
fig_width_in = fig_width_pt * inches_per_pt  # figure width in inches
fig_height_in = fig_width_in * golden_ratio   # figure height in inches
fig_dims = [fig_width_in, fig_height_in]  # fig dims as a list
# Line, marker, and font options
pfont = 11
lw = 1
ms=8
# Begin figure
plt.figure(figsize=fig_dims)


# EFFECTIVE ORDER IN SPACE
# Set up dx array
dx_ary = np.array([0.003125,0.00625, 0.0125, 0.025, 0.05, 0.1])
dt = 0.005
xlog=np.logspace(-2.75, -0.75, 201)

# Calculate RMS for different dx values using trapezoidal method
RMS_ary = np.zeros(dx_ary.size)
for i in range(0,dx_ary.size):
    dx = dx_ary[i]
    t, Phi_T = solution_method.trapezoidal(C = u*dt/dx,s = D*dt/(dx*dx))
    RMS = rms_calc(t, Phi_T)
    RMS_ary[i] = RMS
# Plot RMS values as points and fit a line to them    
plt.plot(dx_ary,RMS_ary,'o',c='b',zorder=2,markersize=ms)
slope_T, intercept_T, _, _, _ = stats.linregress(np.log10(dx_ary),np.log10(RMS_ary))
plt.plot(xlog,np.power(xlog,slope_T)*np.power(10,intercept_T),c='b',linewidth=lw,zorder=1)
plt.xscale('log')
plt.yscale('log')

# Calculate RMS for different dx values using central diff. method
RMS_ary = np.zeros(dx_ary.size)
for i in range(0,dx_ary.size):
    dx = dx_ary[i]
    t, Phi_C = solution_method.central_diff(C = u*dt/dx,s = D*dt/(dx*dx))
    RMS = rms_calc(t, Phi_C)
    RMS_ary[i] = RMS
# Plot RMS values as points and fit a line to them    
plt.plot(dx_ary,RMS_ary,'o',c='g',zorder=2,markersize=ms)
slope_C, intercept_C, _, _, _ = stats.linregress(np.log10(dx_ary),np.log10(RMS_ary))
plt.plot(xlog,np.power(xlog,slope_C)*np.power(10,intercept_C),c='g',linewidth=lw,zorder=1)
plt.xscale('log')
plt.yscale('log')

# Calculate RMS for different dx values using upwind method
RMS_ary = np.zeros(dx_ary.size)
for i in range(0,dx_ary.size):
    dx = dx_ary[i]
    t, Phi_U = solution_method.upwind(C = u*dt/dx,s = D*dt/(dx*dx))
    RMS = rms_calc(t, Phi_U)
    RMS_ary[i] = RMS
# Plot RMS values as points and fit a line to them    
plt.plot(dx_ary,RMS_ary,'o',c='r',zorder=2,markersize=ms)
slope_U, intercept_U, _, _, _ = stats.linregress(np.log10(dx_ary),np.log10(RMS_ary))
plt.plot(xlog,np.power(xlog,slope_U)*np.power(10,intercept_U),c='r',linewidth=lw,zorder=1)
plt.xscale('log')
plt.yscale('log')

# Calculate RMS for different dx values using QUICK method
RMS_ary = np.zeros(dx_ary.size)
for i in range(0,dx_ary.size):
    dx = dx_ary[i]
    t, Phi_Q = solution_method.quick(C = u*dt/dx,s = D*dt/(dx*dx))
    RMS = rms_calc(t, Phi_Q)
    RMS_ary[i] = RMS
# Plot RMS values as points and fit a line to them    
plt.plot(dx_ary,RMS_ary,'o',c='c',zorder=2,markersize=ms)
slope_Q, intercept_Q, _, _, _ = stats.linregress(np.log10(dx_ary),np.log10(RMS_ary))
plt.plot(xlog,np.power(xlog,slope_Q)*np.power(10,intercept_Q),c='c',linewidth=lw,zorder=1)
plt.xscale('log')
plt.yscale('log')

# Plot labels
plt.title('Effective Order in Space',fontsize=pfont+1)
plt.xlabel('$\Delta x$',fontsize=pfont+3,labelpad=0)
plt.ylabel('RMS',fontsize=pfont,labelpad=0)
plt.tick_params(labelsize=pfont)
plt.legend(['Trap.','EO=%.2f' %slope_T,'Cen. Diff.','EO=%.2f' %abs(slope_C),'Upwind','EO=%.2f' %abs(slope_U),'QUICK','EO=%.2f' %abs(slope_Q)],loc='lower center',fontsize=pfont,ncol=4,columnspacing=1.2)
# Save plot as eps
plt.savefig('EffectiveOrderSpace.eps', format='eps',bbox_inches='tight')


# EFFECTIVE ORDER IN TIME
# Set up dt array
dx = 0.005
dt_ary = np.array([0.03125, 0.0625, 0.125 ,0.25, 0.5, 1])
xlog=np.logspace(-1.75, 0.25, 201)

# Calculate RMS for different dt values using trapezoidal method
RMS_ary = np.zeros(dt_ary.size)
for i in range(0,dt_ary.size):
    dt = dt_ary[i]
    t, Phi_T = solution_method.trapezoidal(C = u*dt/dx,s = D*dt/(dx*dx))
    RMS = rms_calc(t, Phi_T)
    RMS_ary[i] = RMS
# Plot RMS values as points and fit a line to them
plt.figure(figsize=fig_dims)
plt.plot(dt_ary,RMS_ary,'o',c='b',zorder=2,markersize=ms)
slope_T, intercept_T, _, _, _ = stats.linregress(np.log10(dt_ary),np.log10(RMS_ary))
plt.plot(xlog,np.power(xlog,slope_T)*np.power(10,intercept_T),c='b',linewidth=lw,zorder=1)
plt.xscale('log')
plt.yscale('log')

# Calculate RMS for different dt values using central diff. method
RMS_ary = np.zeros(dt_ary.size)
for i in range(0,dt_ary.size):
    dt = dt_ary[i]
    t, Phi_T = solution_method.central_diff(C = u*dt/dx,s = D*dt/(dx*dx))
    RMS = rms_calc(t, Phi_T)
    RMS_ary[i] = RMS
# Plot RMS values as points and fit a line to them
plt.plot(dt_ary,RMS_ary,'o',c='g',zorder=2,markersize=ms)
slope_C, intercept_C, _, _, _ = stats.linregress(np.log10(dt_ary),np.log10(RMS_ary))
plt.plot(xlog,np.power(xlog,slope_C)*np.power(10,intercept_C),c='g',linewidth=lw,zorder=1)
plt.xscale('log')
plt.yscale('log')

# Calculate RMS for different dt values using upwind method
RMS_ary = np.zeros(dt_ary.size)
for i in range(0,dt_ary.size):
    dt = dt_ary[i]
    t, Phi_T = solution_method.upwind(C = u*dt/dx,s = D*dt/(dx*dx))
    RMS = rms_calc(t, Phi_T)
    RMS_ary[i] = RMS
# Plot RMS values as points and fit a line to them
plt.plot(dt_ary,RMS_ary,'o',c='r',zorder=2,markersize=ms)
slope_U, intercept_U, _, _, _ = stats.linregress(np.log10(dt_ary),np.log10(RMS_ary))
plt.plot(xlog,np.power(xlog,slope_U)*np.power(10,intercept_U),c='r',linewidth=lw,zorder=1)
plt.xscale('log')
plt.yscale('log')

# Calculate RMS for different dt values using QUICK method
RMS_ary = np.zeros(dt_ary.size)
for i in range(0,dt_ary.size):
    dt = dt_ary[i]
    t, Phi_T = solution_method.quick(C = u*dt/dx,s = D*dt/(dx*dx))
    RMS = rms_calc(t, Phi_T)
    RMS_ary[i] = RMS
# Plot RMS values as points and fit a line to them
plt.plot(dt_ary,RMS_ary,'o',c='c',zorder=2,markersize=ms)
slope_Q, intercept_Q, _, _, _ = stats.linregress(np.log10(dt_ary),np.log10(RMS_ary))
plt.plot(xlog,np.power(xlog,slope_Q)*np.power(10,intercept_Q),c='c',linewidth=lw,zorder=1)
plt.xscale('log')
plt.yscale('log')

# Plot labels
plt.legend(['Trap.','EO=%.2f' %slope_T,'Cen. Diff.','EO=%.2f' %abs(slope_C),'Upwind','EO=%.2f' %abs(slope_U),'QUICK','EO=%.2f' %abs(slope_Q)],loc='lower center',fontsize=pfont,ncol=4,columnspacing=1.2)
plt.axis([1e-2, 1e1, 1e-6, 1e0])
plt.title('Effective Order in Time',fontsize=pfont+1)
plt.xlabel('$\Delta t$',fontsize=pfont+3,labelpad=0)
plt.ylabel('RMS',fontsize=pfont,labelpad=0)
plt.tick_params(labelsize=pfont)
# Save plot as eps
plt.savefig('EffectiveOrderTime.eps', format='eps',bbox_inches='tight')