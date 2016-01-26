# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:33:59 2015

@author: Wolfer
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

from scipy.integrate import ode

from numpy import array

def Fct_sys(t,Phi,M):    
    Phi_dot = np.dot(M,Phi).T    
    return Phi_dot

def solve_dydt(tSpan,y0,options,M):
    
    # unpack option values
    opt_rtol = options[0]
    opt_atol = options[1]
    dt = options[2]
    # use initial and final time range
    t0 = tSpan[0]
    tmax = tSpan[1]
    # set equation to solve
    solver = ode(Fct_sys)
    # initialize time and solution vectors
    t   = []
    sol = []
    sol.append(y0)
    t.append(t0)
    # solver initial value
    solver.set_initial_value(y0, t0).set_f_params(M)
    # solver options
    solver.set_integrator('dopri5',atol=opt_atol,rtol=opt_rtol)

    # Start solver 
    while solver.successful() and solver.t < tmax:
        solver.integrate(solver.t + dt)
        t.append(solver.t)
        sol.append(solver.y)             
    
    # Array conversion and processing for sensitivity and plotting
    t   = array(t)
    sol = array(sol)

    return t, sol

def Analyitical(tau,Phi):
    # Constants
    L = 1.              # m
    D = 0.005           # m^2/s
    u = 0.2             # m/s
    k = 2*np.pi/L       # m^-1
    #tau = 1/((k**2)*D)
    x = np.linspace(0,L,Phi.size)
    Phi_A=np.exp(-k ** 2 * D * tau) * np.sin(k * (x - u * tau))
    return Phi_A
    
def RMS_calc(t,Phi):
    N = Phi.size
    Phi_A = Analyitical(t,Phi)
    rmsum = 0.0
    for i in range(0,N):
        rmsum += (Phi[i] - Phi_A[i])**2
    RMS = np.sqrt((1.0/N)*rmsum)
    return RMS

def Trapezoidal(dx,dt):
    # Constants
    L = 1.              # m
    D = 0.005           # m^2/s
    u = 0.2             # m/s
    k = 2*np.pi/L       # m^-1
    tau = 1/((k**2)*D)
    
    C = u*dt/dx
    s = D*dt/(dx*dx)
    
    x = np.append(np.arange(0,L,dx),L)
    N = x.size
    
    # Initial Condition
    Phi_old = np.sin(k*x)
    
    a = -s/2. - C/4.
    b = 1. + s
    c = -s/2. + C/4.
    
    # Coefficient matrix M
    # N nodes, but N-1 unknowns and eqs
    # because of periodic B.C.
    M=np.zeros((N-1,N-1))
    M[0,0] = b
    M[0,1] = c
    M[0,-1] = a
    M[-1,0] = c
    M[-1,-2] = a
    M[-1,-1] = b
    for i in range(1,N-2):
        M[i,i-1] = a
        M[i,i] = b
        M[i,i+1] = c
    
    rhs=np.zeros(N-1)
    time = 0.0    
    while time <= tau:
        
        rhs[0] = -a*Phi_old[-2] + (2-b)*Phi_old[0] - c*Phi_old[1]
        
        for i in range(1,N-1):
            rhs[i] = -a*Phi_old[i-1] + (2-b)*Phi_old[i] - c*Phi_old[i+1]
        
        # Solve matrix M
        Phi = np.linalg.solve(M, rhs)
        # Adding on last node, equal to first node
        Phi = np.insert(Phi,N-1,Phi[0])

        time = time + dt
        Phi_old = np.copy(Phi)

    return time, Phi
    

def Central_Diff(dx,dt): 
    # Constants
    L = 1.              # m
    D = 0.005           # m^2/s
    u = 0.2             # m/s
    k = 2*np.pi/L       # m^-1
    tau = 1/((k**2)*D)
    
    C = u*dt/dx
    s = D*dt/(dx*dx)
    
    x = np.append(np.arange(0,L,dx),L)
    N = x.size
    
    # Initial Condition
    Phi_old = np.sin(k*x)    
    
    # For Central Differencing
    a = u/(2*dx) + D/(dx*dx)
    b = -2*D/(dx*dx)
    c = -u/(2*dx) + D/(dx*dx)
    
    # Coefficient matrix M
    # N nodes, but N-1 unknowns and eqs
    # because of periodic B.C.
    M=np.zeros((N-1,N-1))
    M[0,0] = b
    M[0,1] = c
    M[0,-1] = a
    M[-1,0] = c
    M[-1,-2] = a
    M[-1,-1] = b
    for i in range(1,N-2):
        M[i,i-1] = a
        M[i,i] = b
        M[i,i+1] = c
    
    # solver options
    options = [1e-6, 1e-6, dt]
    # time span
    tspan = [0, tau]
    # initial (chopping off last point because periodic B.C.)
    y0 = Phi_old[0:-1]
    # call ode45 solver funciton
    tout, yout = solve_dydt(tspan, y0, options,M);
    
    tend = tout[-1]
    
    Phi = yout[-1,:]
    Phi = np.append(Phi,Phi[0])
    
    return tend, Phi
    
def Upwind(dx,dt): 
    # Constants
    L = 1.              # m
    D = 0.005           # m^2/s
    u = 0.2             # m/s
    k = 2*np.pi/L       # m^-1
    tau = 1/((k**2)*D)
    
    C = u*dt/dx
    s = D*dt/(dx*dx)
    
    x = np.append(np.arange(0,L,dx),L)
    N = x.size
    
    # Initial Condition
    Phi_old = np.sin(k*x)    
    
    # For Upwinding 1st order
    a = u/(dx) + D/(dx*dx)
    b = -u/(dx) - 2*D/(dx*dx)
    c = D/(dx*dx)
    
    # Coefficient matrix M
    M=np.zeros((N-1,N-1))
    M[0,0] = b
    M[0,1] = c
    M[0,-1] = a
    M[-1,0] = c
    M[-1,-2] = a
    M[-1,-1] = b
    for i in range(1,N-2):
        M[i,i-1] = a
        M[i,i] = b
        M[i,i+1] = c
    
#    # For Upwinding 2nd order
#    d = -u/(2*dx)
#    a = 4*u/(2*dx) + D/(dx*dx)
#    b = -3*u/(2*dx) - 2*D/(dx*dx)
#    c = D/(dx*dx)    
#    # Coefficient matrix M
#    M=np.zeros((N-1,N-1))
#    M[0,0] = b
#    M[0,1] = c
#    M[0,-1] = a
#    M[0,-2] = d
#    M[1,0] = a
#    M[1,1] = b
#    M[1,2] = c
#    M[1,-1] = d
#    M[-1,0] = c
#    M[-1,-3] = d
#    M[-1,-2] = a
#    M[-1,-1] = b
#    for i in range(2,N-2):
#        M[i,i-2] = d
#        M[i,i-1] = a
#        M[i,i] = b
#        M[i,i+1] = c
    
    # solver options
    options = [1e-6, 1e-6, dt]
    # time span
    tspan = [0, tau]
    # initial (chopping off last point because periodic B.C.)
    y0 = Phi_old[0:-1]
    # call ode45 solver funciton
    
    tout, yout = solve_dydt(tspan, y0, options,M);
    tend = tout[-1]
    
    Phi = yout[-1,:]
    Phi = np.append(Phi,Phi[0])
    
    return tend, Phi  

   
def Quick(dx,dt): 
    # Constants
    L = 1.              # m
    D = 0.005           # m^2/s
    u = 0.2             # m/s
    k = 2*np.pi/L       # m^-1
    tau = 1/((k**2)*D)
    
    C = u*dt/dx
    s = D*dt/(dx*dx)
    
    x = np.append(np.arange(0,L,dx),L)
    N = x.size
    
    # Initial Condition
    Phi_old = np.sin(k*x)    
     
    # Quick 
    d = -u/(8*dx)
    a = 7*u/(8*dx) + D/(dx*dx)
    b = -3*u/(8*dx) - 2*D/(dx*dx)
    c = -3*u/(8*dx) + D/(dx*dx)    
    # Coefficient matrix M
    M=np.zeros((N-1,N-1))
    M[0,0] = b
    M[0,1] = c
    M[0,-1] = a
    M[0,-2] = d
    M[1,0] = a
    M[1,1] = b
    M[1,2] = c
    M[1,-1] = d
    M[-1,0] = c
    M[-1,-3] = d
    M[-1,-2] = a
    M[-1,-1] = b
    for i in range(2,N-2):
        M[i,i-2] = d
        M[i,i-1] = a
        M[i,i] = b
        M[i,i+1] = c

    # solver options
    options = [1e-6, 1e-6, dt]
    # time span
    tspan = [0, tau]
    # initial (chopping off last point because periodic B.C.)
    y0 = Phi_old[0:-1]
    # call ode45 solver funciton
    tout, yout = solve_dydt(tspan, y0, options,M);
    tend = tout[-1]
    
    Phi = yout[-1,:]
    Phi = np.append(Phi,Phi[0])
    
    return tend, Phi  


# ***************** BEGIN CALLING ABOVE FUNCTIONS ****************************


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

lw = 1
ms=8

plt.figure(figsize=fig_dims)



dx_ary = np.array([0.003125,0.00625, 0.0125, 0.025, 0.05, 0.1])

dt = 0.005

xlog=np.logspace(-2.75, -0.75, 201)

RMS_ary = np.zeros(dx_ary.size)

for i in range(0,dx_ary.size):

    dx = dx_ary[i]

    t, Phi_T = Trapezoidal(dx,dt)
    RMS = RMS_calc(t, Phi_T)
    RMS_ary[i] = RMS
    print RMS
    
    

plt.plot(dx_ary,RMS_ary,'o',c='b',zorder=2,markersize=ms)
slope_T, intercept_T, _, _, _ = stats.linregress(np.log10(dx_ary),np.log10(RMS_ary))
plt.plot(xlog,np.power(xlog,slope_T)*np.power(10,intercept_T),c='b',linewidth=lw,zorder=1)

plt.xscale('log')
plt.yscale('log')


RMS_ary = np.zeros(dx_ary.size)

for i in range(0,dx_ary.size):

    dx = dx_ary[i]

    t, Phi_C = Central_Diff(dx,dt)
    RMS = RMS_calc(t, Phi_C)
    RMS_ary[i] = RMS
    print RMS

plt.plot(dx_ary,RMS_ary,'o',c='g',zorder=2,markersize=ms)
slope_C, intercept_C, _, _, _ = stats.linregress(np.log10(dx_ary),np.log10(RMS_ary))
plt.plot(xlog,np.power(xlog,slope_C)*np.power(10,intercept_C),c='g',linewidth=lw,zorder=1)

plt.xscale('log')
plt.yscale('log')


RMS_ary = np.zeros(dx_ary.size)

for i in range(0,dx_ary.size):

    dx = dx_ary[i]

    t, Phi_U = Upwind(dx,dt)
    RMS = RMS_calc(t, Phi_U)
    RMS_ary[i] = RMS

plt.plot(dx_ary,RMS_ary,'o',c='r',zorder=2,markersize=ms)
slope_U, intercept_U, _, _, _ = stats.linregress(np.log10(dx_ary),np.log10(RMS_ary))
plt.plot(xlog,np.power(xlog,slope_U)*np.power(10,intercept_U),c='r',linewidth=lw,zorder=1)

plt.xscale('log')
plt.yscale('log')


RMS_ary = np.zeros(dx_ary.size)

for i in range(0,dx_ary.size):

    dx = dx_ary[i]

    t, Phi_Q = Quick(dx,dt)
    RMS = RMS_calc(t, Phi_Q)
    RMS_ary[i] = RMS

plt.plot(dx_ary,RMS_ary,'o',c='c',zorder=2,markersize=ms)
slope_Q, intercept_Q, _, _, _ = stats.linregress(np.log10(dx_ary),np.log10(RMS_ary))
plt.plot(xlog,np.power(xlog,slope_Q)*np.power(10,intercept_Q),c='c',linewidth=lw,zorder=1)

plt.xscale('log')
plt.yscale('log')

plt.title('Effective Order in Space',fontsize=pfont+1)
plt.xlabel('$\Delta x$',fontsize=pfont+3,labelpad=0)
plt.ylabel('RMS',fontsize=pfont,labelpad=0)
plt.tick_params(labelsize=pfont)
plt.legend(['Trap.','EO=%.2f' %slope_T,'Cen. Diff.','EO=%.2f' %abs(slope_C),'Upwind','EO=%.2f' %abs(slope_U),'QUICK','EO=%.2f' %abs(slope_Q)],loc='lower center',fontsize=pfont,ncol=4,columnspacing=1.2)

plt.savefig('EffectiveOrderSpace.eps', format='eps',bbox_inches='tight')









xlog=np.logspace(-1.75, 0.25, 201)

dx = 0.005

#dt_ary = np.array([0.005, 0.03125, 0.0625, 0.125, 1, 2, 0.01, 0.25, 0.5, 0.001])

dt_ary = np.array([0.03125, 0.0625, 0.125 ,0.25, 0.5, 1])

RMS_ary = np.zeros(dt_ary.size)

for i in range(0,dt_ary.size):

    dt = dt_ary[i]

    t, Phi_T = Trapezoidal(dx,dt)
    RMS = RMS_calc(t, Phi_T)
    RMS_ary[i] = RMS

plt.figure(figsize=fig_dims)
plt.plot(dt_ary,RMS_ary,'o',c='b',zorder=2,markersize=ms)
slope_T, intercept_T, _, _, _ = stats.linregress(np.log10(dt_ary),np.log10(RMS_ary))
plt.plot(xlog,np.power(xlog,slope_T)*np.power(10,intercept_T),c='b',linewidth=lw,zorder=1)
plt.xscale('log')
plt.yscale('log')


RMS_ary = np.zeros(dt_ary.size)

for i in range(0,dt_ary.size):

    dt = dt_ary[i]

    t, Phi_T = Central_Diff(dx,dt)
    RMS = RMS_calc(t, Phi_T)
    RMS_ary[i] = RMS

plt.plot(dt_ary,RMS_ary,'o',c='g',zorder=2,markersize=ms)
slope_C, intercept_C, _, _, _ = stats.linregress(np.log10(dt_ary),np.log10(RMS_ary))
plt.plot(xlog,np.power(xlog,slope_C)*np.power(10,intercept_C),c='g',linewidth=lw,zorder=1)
plt.xscale('log')
plt.yscale('log')


RMS_ary = np.zeros(dt_ary.size)

for i in range(0,dt_ary.size):

    dt = dt_ary[i]

    t, Phi_T = Upwind(dx,dt)
    RMS = RMS_calc(t, Phi_T)
    RMS_ary[i] = RMS

plt.plot(dt_ary,RMS_ary,'o',c='r',zorder=2,markersize=ms)
slope_U, intercept_U, _, _, _ = stats.linregress(np.log10(dt_ary),np.log10(RMS_ary))
plt.plot(xlog,np.power(xlog,slope_U)*np.power(10,intercept_U),c='r',linewidth=lw,zorder=1)

plt.xscale('log')
plt.yscale('log')


RMS_ary = np.zeros(dt_ary.size)

for i in range(0,dt_ary.size):

    dt = dt_ary[i]

    t, Phi_T = Quick(dx,dt)
    RMS = RMS_calc(t, Phi_T)
    RMS_ary[i] = RMS

plt.plot(dt_ary,RMS_ary,'o',c='c',zorder=2,markersize=ms)
slope_Q, intercept_Q, _, _, _ = stats.linregress(np.log10(dt_ary),np.log10(RMS_ary))
plt.plot(xlog,np.power(xlog,slope_Q)*np.power(10,intercept_Q),c='c',linewidth=lw,zorder=1)
plt.xscale('log')
plt.yscale('log')


plt.legend(['Trap.','EO=%.2f' %slope_T,'Cen. Diff.','EO=%.2f' %abs(slope_C),'Upwind','EO=%.2f' %abs(slope_U),'QUICK','EO=%.2f' %abs(slope_Q)],loc='lower center',fontsize=pfont,ncol=4,columnspacing=1.2)



plt.axis([1e-2, 1e1, 1e-6, 1e0])

plt.title('Effective Order in Time',fontsize=pfont+1)
plt.xlabel('$\Delta t$',fontsize=pfont+3,labelpad=0)
plt.ylabel('RMS',fontsize=pfont,labelpad=0)
plt.tick_params(labelsize=pfont)

plt.savefig('EffectiveOrderTime.eps', format='eps',bbox_inches='tight')
