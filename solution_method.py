# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:31:51 2015

@author: Wolfer
"""

import numpy as np
from my_ode import solve_dydt

def analyitical(tau,Phi):
    # Constants
    L = 1.              # m
    D = 0.005           # m^2/s
    u = 0.2             # m/s
    k = 2*np.pi/L       # m^-1
    #tau = 1/((k**2)*D)
    x = np.linspace(0,L,Phi.size)
    Phi_A=np.exp(-k ** 2 * D * tau) * np.sin(k * (x - u * tau))
    return Phi_A

def trapezoidal(C,s):
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
    
def trapezoidal_alt(C,s):
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

def central_diff(C,s): 
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
    
def upwind(C,s): 
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

   
def quick(C,s): 
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


