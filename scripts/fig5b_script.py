"""
Script to produce figure 5(b) in the paper.
"""

import numpy as np 
import matplotlib.pyplot as plt
from models.network import VSM 
from scipy.integrate import solve_ivp
import warnings
from models.ideal import LCLRsinglephase 
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

# Range of parameter values to calculate for
Js = [1.0, 9.0]

# Find critical points
crit_points = []
for J in Js:
    try:
        crit_R = find_critical_point(VSM, J)
    except ArithmeticError:
        break
    crit_points.append(crit_R)
    print(f"At J = {J}, criticality at {crit_R}")


# Iterate over the J values, and plot R vs n for each
for i, j in enumerate(Js):
    r = crit_points[i]

    # Get the critical curve for this R_crit
    nvec, Pvec, Rvec = compute_power_curve(r)

    # Plot
    plt.plot(Rvec, nvec, label=f"J = {j}")

plt.legend(loc=1)
plt.xlim([100,500])
plt.ylim([10,50])
plt.xlabel("R_d")
plt.ylabel("n")
plt.show()