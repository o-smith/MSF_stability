"""
Script to produce figure 5(d,e,f) in the paper.
"""

import numpy as np 
import matplotlib.pyplot as plt
from models.network import VSM, DroopDroop
from scipy.integrate import solve_ivp
from models.ideal import LCLRsinglephase 
import warnings
warnings.filterwarnings("ignore")


def find_critical_point(model: object, J: float) -> float:
    """
    Function to find the point at which the MSF crosses the axis.
    """

    # Set up model
    ob = model() 
    ob.n = 1
    ob.Rload =  50.0
    ob.J = J
    x0 = np.zeros(ob.n*ob.m)
    min_delta_R = 0.1
    delta_R = 1000.0
    ob.initialise_system() 

    # Bisect to find unstable point
    previously_stable = True
    while abs(delta_R) > min_delta_R:

        #Time-step
        ob.initialise_system() 
        sol = solve_ivp(ob.F, [0,2], x0, rtol=1e-10, method="LSODA", min_step=10e-8)

        # Check if steady state found
        if sol.status == 0: 
            stable = True 
        else: 
            stable = False
        
        # If there has been a change in stability
        if stable != previously_stable:
            delta_R = -delta_R/2.0

        # Update
        previously_stable = stable
        ob.Rload += delta_R
        print(ob.Rload)
        if ob.Rload <= 0.0:
            raise ArithmeticError

    return ob.Rload

def compute_power_curve(
        R_crit: float,
        nmax: int = 20000
        ) -> tuple[list, list, list]:
    """
    Function to compute n vs P stability curve and 
    n vs R stability curve.

    Input
    -----
    R_crit: float
        The critical point at which MSF crosses axis.
    nmax: int
        The maximum number of nodes to compute for.

    Outputs
    -------
    n, P, R: list, list, list
        P and R as functions of n.
    """
    # Set up steady state system for power calculations
    inverter = LCLRsinglephase() 
    inverter.Ld = 1e-3
    R_transmission = 0.06
    P, Rs = [], []
    ns = np.arange(1,nmax,1)
    for n in ns:
        R = R_crit/n 
        Rs.append(R)
        inverter.R = R + R_transmission
        P.append(3*inverter.real_power())
    return ns, P, Rs

# Find critical point for droop controller
crit_R_Droop = find_critical_point(DroopDroop, 1.0)

# Find critical point for regular VSM
crit_R_VSM = find_critical_point(VSM, 1.0)

# Compute the power curve for J = 1
nvec, Pvec, Rvec = compute_power_curve(crit_R_VSM)

# Compute the power curve for the droop controller
nvec_droop, Pvec_droop, Rvec_droop = compute_power_curve(crit_R_Droop)

# Compute the power value corresponding to R_d = 36.9
# This is the power value at which there are over-currents
# as computed in the appendix
inverter = LCLRsinglephase() 
inverter.Ld = 1e-3
R_transmission = 0.06
inverter.R = 36.9 + R_transmission
cutoff = 3*inverter.real_power()

# Plot the curve and find good scale for it
plt.plot(Pvec, nvec)
plt.plot(Pvec_droop, nvec_droop)
plt.fill_between(Pvec, nvec, where=((Pvec <= cutoff) & (nvec <=200)), color='lightblue', alpha=0.5)
plt.plot(1297, 6, "o", color="k")
plt.plot(85.4, 6, "o", color="k")
plt.ylim([0,200])
plt.xlim([0,2350])
plt.axvline(cutoff, color="k", linestyle="--")
plt.xlabel("P")
plt.ylabel("n")
plt.show()
