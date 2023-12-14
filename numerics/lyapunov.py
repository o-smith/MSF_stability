#! usr/bin/env python 

"""
File containing functions to compute various things of interest for the master stability function:
-- Lyapunov exponents as a function of time (compute_lyapunov)
-- Asymptotic value of largest lyapunov exponent, from a time series (lyapunov_extrapolation)
-- A bisection method to compute approximate location of zero-crossing point of MSF (lyapunov_bisection)
"""

import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def compute_lyapunov(f, fjac, hess, x0, t, ttrans, QRfreq, rtol=1e-10, noisy=False, solver="RK45", extrastiff=False):
    """
    Function to compute the maximum Lyapunov exponent of a system of ODEs f
    """

    n = len(x0)
    
    def dM_dt(t, M, x):
        """ The variational equation """
        Mmat = np.reshape(M, (n,n))
        dMmat = np.dot(fjac(t, x), Mmat)
        return dMmat.flatten()

    def dpsi_dt(t, psi):
        """
        Differential equations for combined state/variational matrix
        propagation. This combined state is called S.
        """
        x = psi[:n]
        M = psi[n:]
        return np.append(f(t,x), dM_dt(t, M, x))

    # integrate transient behavior 
    print("Finding attractor ...") 
    if extrastiff:
        sol = solve_ivp(f, [0,ttrans], x0, rtol=1e-10, method="LSODA") #, jac=hess)
    else:
        sol = solve_ivp(f, [0,ttrans], x0, rtol=rtol)

    # for i in range(12):
    #     plt.plot(sol.t, sol.y[i,:])
    # plt.show()

    #Initial condition for Lyapunov integration 
    x0 = sol.y[:,-1]
    M0 = np.eye(n, dtype=np.float64).flatten()
    psi0 = np.append(x0, M0) 

    #Time points for Lyapunov integration
    trange = np.arange(0, t, QRfreq) 

    #Structure to store multipliers for each QR step 
    rs = np.zeros((len(trange)-1, n)) 

    #Iterate over the time points. Solve ivp between each pair of 
    #of points and do QR re-orthogonalisation 
    print("Beginning Lyapunov calculations ... ")
    for i in range(len(trange) - 1): 

        if noisy:
            print("t = ", trange[i]) 

        #Integrate 
        if hess != None:
            sol = solve_ivp(dpsi_dt, [trange[i],trange[i+1]], psi0, rtol=rtol, method=solver, jac=hess) 
        else:
            sol = solve_ivp(dpsi_dt, [trange[i],trange[i+1]], psi0, rtol=rtol, method=solver) 

        #Re-orthonormalise using QR decomposition 
        psi = sol.y[:,-1]
        M = np.reshape(psi[n:], (n,n)) 
        Q, R = np.linalg.qr(M) 

        #Construct the intial value for next integration
        psi0[:n] = psi[:n]
        psi0[n:] = Q.flatten() 

        #Store the multipliers 
        rs[i,:] = np.abs(np.diag(R)) 

    lyapunov_exponents = np.cumsum(np.log(rs), axis=0)/np.tile(trange[1:], (n,1)).T 
    return lyapunov_exponents.max(axis=1)


def lyapunov_extrapolation(fname, cutoff, tendcutoff, t=None, lam=None, extension=5000, plot=True):
    """
    Function to take the time-series of the Lyapunov exponent
    and interpolate it further in time, if possible. 
    """

    def func(x, a, b, c):
        """Fit to 1/x""" 
        return a/(x-c) + b 

    if fname != None:
        z = np.genfromtxt(fname)
        t, lam = np.hsplit(z, 2) 
        istart = np.where(t>cutoff)[0][0] 
        ifin = np.where(t<tendcutoff)[0][-1]
        tp = np.array(t[istart:ifin])[:,0]
        lamp = np.array(lam[istart:ifin])[:,0]
    else:
        istart = np.where(t>cutoff)[0][0]
        ifin = np.where(t<tendcutoff)[0][-1]
        tp = np.array(t[istart:ifin])
        lamp = np.array(lam[istart:ifin])

    popt, pcov = curve_fit(func, tp, lamp)
    print("Fit patameters:", popt)
    print("Standard deviation:", np.sqrt(np.diag(pcov)))

    LEmax = func(extension, *popt)
    if plot:
        plt.plot(tp, lamp, "m-", lw=4) 
        plt.plot(tp, func(tp, *popt), "c-", lw=3)
        plt.plot(tp, func(tp, *popt))
        plt.show() 
    return LEmax 


def lyapunov_bisection(ob, pname, pstartval, x0, bitol=1e-1, tmax=3): 
    """
    Function to find approximate location of the zero crossing point of the 
    master stability function using a bisection of the steady state equation. 
    """

    stepsize = 100 
    converged_old = True 
    while abs(stepsize) > bitol:
        print(pstartval, stepsize)

        setattr(ob, pname, pstartval)
        if hasattr(ob, "initialise_system"):
            ob.initialise_system() 

        sol = solve_ivp(ob.Fsteady, [0,tmax], x0, rtol=1e-13, method="LSODA", min_step=1e-8)  

        # Has there been a change in success since last step? 
        if converged_old == sol.success: 
            pstartval += stepsize 
        else: 
            stepsize = -stepsize/2 
            pstartval += stepsize 
        converged_old = sol.success 
    
    if converged_old == False:
        return pstartval - stepsize 
    else:
        return pstartval 
        