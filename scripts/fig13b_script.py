"""
Script to produce figure 13b in the paper.
"""

import numpy as np 
import matplotlib.pyplot as plt
from models.network import VSM_complexload
from scipy.integrate import solve_ivp
from models.ideal import LCLRsinglephase
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore")


def find_critical_point(model: object, Ld: float) -> float:
    """
    Function to find the point at which the MSF crosses the axis.
    """

    # Set up model
    ob = model() 
    ob.n = 1
    ob.Rload =  50.0
    ob.Ld = Ld
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

crit_R_Ld0 = find_critical_point(VSM_complexload, 0.0)
crit_R_Ld001 = find_critical_point(VSM_complexload, 0.001)
crit_R_Ld002 = find_critical_point(VSM_complexload, 0.002)
crit_R_Ld02 = find_critical_point(VSM_complexload, 0.02)
crit_R_Ld05 = find_critical_point(VSM_complexload, 0.05)
print(f"crit_R_Ld0 is {crit_R_Ld0}")
print(f"crit_R_Ld001 is {crit_R_Ld001}")
print(f"crit_R_Ld002 is {crit_R_Ld002}")
print(f"crit_R_Ld02 is {crit_R_Ld02}")
print(f"crit_R_Ld05 is {crit_R_Ld05}")

# Get the critical curves and plot
nvec1, Pvec1, Rvec1 = compute_power_curve(crit_R_Ld0)
nvec2, Pvec2, Rvec2 = compute_power_curve(crit_R_Ld001)
nvec3, Pvec3, Rvec3 = compute_power_curve(crit_R_Ld002)
nvec4, Pvec4, Rvec4 = compute_power_curve(crit_R_Ld02)
nvec5, Pvec5, Rvec5 = compute_power_curve(crit_R_Ld05)

# Create a figure and main axes
fig, ax = plt.subplots()

# Your original plot
ax.plot(Rvec1, nvec1, label="Ld = 0")
ax.plot(Rvec2, nvec2, label="Ld = 0.001")
ax.plot(Rvec3, nvec3, label="Ld = 0.002")
ax.plot(Rvec4, nvec4, label ="Ld = 0.02")
ax.plot(Rvec5, nvec5, label="Ld = 0.05")
ax.legend(loc=3)
ax.set_xlim([100, 500])
ax.set_ylim([10, 50])
ax.set_xlabel("R_d")
ax.set_ylabel("n")

# Define the zoomed region boundaries
zoom_x_start, zoom_x_end = 254, 263
zoom_y_start, zoom_y_end = 22.4, 23.1

# Create an inset subplot
ax_inset = inset_axes(ax, width="50%", height="50%", loc=1)

# Plot the zoomed region in the inset subplot
ax_inset.plot(Rvec1, nvec1)
ax_inset.plot(Rvec2, nvec2)
ax_inset.plot(Rvec3, nvec3)
ax_inset.plot(Rvec4, nvec4)
ax_inset.plot(Rvec5, nvec5)
ax_inset.set_xlim(zoom_x_start, zoom_x_end)
ax_inset.set_ylim(zoom_y_start, zoom_y_end)
ax_inset.set_xticks([])
ax_inset.set_yticks([])

ax.indicate_inset_zoom(ax_inset, edgecolor="black")



plt.show()


