# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:31:51 2015

@author: Wolfer
"""

import numpy as np
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
    