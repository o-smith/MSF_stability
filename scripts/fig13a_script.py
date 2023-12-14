"""
Script to produce figure ... in the paper.

This script produces the MSF for the VSM with complex load.

WARNING: 
    This script takes a long time to run! It will typically
    tie up a high spec macbook pro for at least 48 hours. This
    is because we are computing two MSFs to a high precision and using
    a long integration time. This script is ONLY used to compute
    the full MSF for illustrative purposes. In general, we are 
    only concerned with the critical point where the MSF becomes
    positive. This is found very much faster (order seconds or
    minutes) in fig13b_script.py.

Critical values are [5780.89599609375, 5961.07177734375] for
Ld = 0mH, 2mH respectively.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from models.network import VSM_complexload
from numerics.lyapunov import compute_lyapunov, lyapunov_extrapolation

# Check directory for saving the results into exists
fdir = os.path.join("data", "paper", "13a")
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
Rloads = np.linspace(100,5960,20)

# First, compute the MSF for Ld = 0 case
# Iterate over the load values
for R in Rloads:

    print(f"Computing Lyapunov exponents for Rload = {R}")

    # Create VSM instance
    ob = VSM_complexload()
    ob.n = 1
    ob.Ld = 0.0
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
    fname = os.path.join(fdir_abs, f"Ld_0_R_{R}.txt")

    # Save the data to the directory
    t = np.linspace(0, tmax, len(LE))
    np.savetxt(fname, list(zip(t, LE)))


# Post-process and plot the MSF
le = []
for r in Rloads:
    fname = os.path.join(fdir_abs, f"Ld_0_R_{R}.txt")
    print(fname)
    LE_limit_extrap = lyapunov_extrapolation(fname, 40, tmax, extension=10000,plot=False)
    le.append(LE_limit_extrap)
plt.plot(Rloads,le, label="Ld = 0.02")
plt.ylim([min(le)-0.2,0.05])

# Now repeat for the Ld = 0.02 case
# First, compute the MSF for Ld = 0 case
# Iterate over the load values
for R in Rloads:

    print(f"Computing Lyapunov exponents for Rload = {R}")

    # Create VSM instance
    ob = VSM_complexload()
    ob.n = 1
    ob.Ld = 0.002
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
    fname = os.path.join(fdir_abs, f"Ld_002_R_{R}.txt")

    # Save the data to the directory
    t = np.linspace(0, tmax, len(LE))
    np.savetxt(fname, list(zip(t, LE)))

# Post-process and plot the MSF
le = []
for r in Rloads:
    fname = os.path.join(fdir_abs, f"Ld_002_R_{R}.txt")
    print(fname)
    LE_limit_extrap = lyapunov_extrapolation(fname, 40, tmax, extension=10000,plot=False)
    le.append(LE_limit_extrap)
plt.plot(Rloads,le, label="Ld = 0.002")

plt.xlabel("$R_d")
plt.ylabel("Largest Lyapunov")
plt.show()