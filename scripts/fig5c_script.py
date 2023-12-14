"""
Script to produce figure 5(c) in the paper.
"""

import os
import numpy as np 
import matplotlib.pyplot as plt
from models.network import VSM 
from scipy.integrate import solve_ivp
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
        sol = solve_ivp(ob.F, [0,1], x0, rtol=1e-10, method="LSODA", min_step=10e-8)

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


# Range of parameter values to calculate for
Js = np.linspace(10,45,30)

# Find critical points for a range of 
# inertia values
crit_points = []
for J in Js:
    try:
        crit_R = find_critical_point(VSM, J)
    except ArithmeticError:
        break
    crit_points.append(crit_R)
    print(f"At J = {J}, criticality at {crit_R}")

# Plot results
plt.plot(Js,crit_points, "o")
plt.xlabel("J")
plt.ylabel("R_crit")
plt.show()