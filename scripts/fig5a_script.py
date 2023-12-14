"""
Script to produce the figure 5(a) in the paper.

This script produces the MSF for the VSM. 

WARNING: 
    This script takes a long time to run! It will typically
    tie up a high spec macbook pro for at least 24 hours. This
    is because we are computing things to a high precision and using
    a long integration time. This script is ONLY used to compute
    the full MSF for illustrative purposes. In general, we are 
    only concerned with the critical point where the MSF becomes
    positive. This can be found very much faster (order seconds or
    minutes) using a simple bisection method, as used in the 
    script fig5b_script.py and fig5c_script.py.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from models.network import VSM
from numerics.lyapunov import compute_lyapunov, lyapunov_extrapolation

# Check directory for saving the results into exists
fdir = os.path.join("data", "paper", "5a")
fdir_abs = os.path.abspath(fdir)
if os.path.exists(fdir_abs):
    # If it exists, empty its contents
    for root, dirs, files in os.walk(fdir_abs):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
else:
    # If it doesn't exist, create the directory
    os.makedirs(fdir_abs)

# Range of load values at which to evaluate the msf. 
# The critical point for the VSM with the parameters we'll
# be using here is at around 5780. So this range is just 
# to illustrate the general shape of the msf. Other ranges
# could be used for greater resolution
Rloads = [100, 700, 1300, 1990, 2600, 3200, 3800, 4500, 5100, 5780]

# Iterate over the load values
for R in Rloads:

    print(f"Computing Lyapunov exponents for Rload = {R}")

    # Create VSM instance
    ob = VSM()
    ob.n = 1
    ob.omegaks = [2*np.pi*50]
    ob.Rload = R
    ob.initialise_system()

    # Initial state
    x0 = np.zeros(ob.m)

    # Compute largest Lyapunov exponent
    tmax = 200
    LE = compute_lyapunov(ob.Fsteady, ob.variational, ob.jacobian, x0, tmax,20, 0.1, noisy=True, solver="LSODA", extrastiff=True)

    # Handle results by saving lyapunov calculations to a file
    # for each value of Rload. The lyapunov calculations converge
    # slowly as time t goes on
    fname = os.path.join(fdir_abs, f"R_{R}.txt")

    # Save the data to the directory
    t = np.linspace(0, tmax, len(LE))
    np.savetxt(fname, list(zip(t, LE)))


# Post-process and plot the MSF
le = []
for r in Rloads:
    fname = os.path.join(fdir_abs, f"R_{R}.txt")
    print(fname)
    LE_limit_extrap = lyapunov_extrapolation(fname, 40, tmax, extension=10000,plot=False)
    le.append(LE_limit_extrap)
plt.plot(Rloads,le)
plt.ylim([min(le)-0.2,0.05])
plt.xlabel("$R_d")
plt.ylabel("Largest Lyapunov")
plt.show()

