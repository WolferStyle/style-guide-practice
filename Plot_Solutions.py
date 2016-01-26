# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:31:51 2015

@author: Wolfer
"""


import numpy as np
import matplotlib.pyplot as plt

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

def Trapezoidal(C,s):
    # Constants
    L = 1.              # m
    D = 0.005           # m^2/s
    u = 0.2             # m/s
    k = 2*np.pi/L       # m^-1
    tau = 1/((k**2)*D)
    
    dx = C*D/(u*s)
    dt = C*dx/u
    
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
    cnt=1
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
        
        cnt=cnt+1

    print cnt
    return time, Phi
    
def Trapezoidal_ALT(C,s):
    # Constants
    L = 1.              # m
    D = 0.005           # m^2/s
    u = 0.2             # m/s
    k = 2*np.pi/L       # m^-1
    tau = 1/((k**2)*D)
    
    dx = C*D/(u*s)
    dt = C*dx/u
    
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
    M=np.zeros((N,N))
    M[0,0] = b
    M[0,1] = c
    M[0,N-2] = a
    M[N-2,0] = c
    M[N-2,N-3] = a
    M[N-2,N-2] = b
    for i in range(1,N-2):
        M[i,i-1] = a
        M[i,i] = b
        M[i,i+1] = c
    
    M[N-1,0] = 1
    M[N-1,N-1] = -1     
    
    rhs=np.zeros(N)
    time = 0.0    
    while time <= tau:
        
        rhs[0] = -a*Phi_old[-2] + (2-b)*Phi_old[0] - c*Phi_old[1]
        
        for i in range(1,N-1):
            rhs[i] = -a*Phi_old[i-1] + (2-b)*Phi_old[i] - c*Phi_old[i+1]
            
        rhs[N-1] = 0
        
        # Solve matrix M
        Phi = np.linalg.solve(M, rhs)


        time = time + dt
        Phi_old = np.copy(Phi)

    return Phi

def Central_Diff(C,s): 
    # Constants
    L = 1.              # m
    D = 0.005           # m^2/s
    u = 0.2             # m/s
    k = 2*np.pi/L       # m^-1
    tau = 1/((k**2)*D)
    
    dx = C*D/(u*s)
    dt = C*dx/u
    
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
    
def Upwind(C,s): 
    # Constants
    L = 1.              # m
    D = 0.005           # m^2/s
    u = 0.2             # m/s
    k = 2*np.pi/L       # m^-1
    tau = 1/((k**2)*D)
    
    dx = C*D/(u*s)
    dt = C*dx/u
    
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

   
def Quick(C,s): 
    # Constants
    L = 1.              # m
    D = 0.005           # m^2/s
    u = 0.2             # m/s
    k = 2*np.pi/L       # m^-1
    tau = 1/((k**2)*D)
    
    dx = C*D/(u*s)
    dt = C*dx/u
    
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

C_lst = [0.1,   0.5,   2, 0.5, 0.5]
s_lst = [0.25, 0.25, .25, 0.5,   1]

case = 5
C = C_lst[case-1]
s = s_lst[case-1]
dx = C*D/(u*s)
dt = C*dx/u

# Constants
L = 1.              # m
D = 0.005           # m^2/s
u = 0.2             # m/s
k = 2*np.pi/L       # m^-1
tau = 1/((k**2)*D)

#taustar = tau/d

t_T, Phi_T = Trapezoidal(C,s)
Phi_A_T = Analyitical(t_T,Phi_T)

t_C, Phi_C = Central_Diff(C,s)
Phi_A_C = Analyitical(t_C,Phi_C)

t_U, Phi_U = Upwind(C,s)
Phi_A_U = Analyitical(t_U,Phi_U)

t_Q, Phi_Q = Quick(C,s)
Phi_A_Q = Analyitical(t_Q,Phi_Q)

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


plt.plot(np.linspace(0,1,201),Analyitical(tau,np.linspace(0,1,201)),'k',linewidth=lw)
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







