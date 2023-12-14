"""
Script to time-step a full system of VSMs.
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp 
from models.network import VSM

# Initialise system
ob = VSM() 
ob.n = 3
ob.Rload =  2000
ob.initialise_system() 
print("done")

#Time-step 
x0 = np.zeros(ob.n*ob.m)
sol = solve_ivp(ob.F, [0,1.5], x0, rtol=1e-10, method="LSODA", min_step=10e-10)

# Time-step for a higher impedance
ob.Rload =  3000
ob.initialise_system()
x0 = np.zeros(ob.n*ob.m)
sol2 = solve_ivp(ob.F, [0,1.5], x0, rtol=1e-10, method="LSODA", min_step=10e-10)


# # Plot timeseries
fig, axes = plt.subplots(1,2)
for i in range(ob.m):
    axes[0].plot(sol.t, sol.y[i,:], alpha=1, lw=2.5)
for i in range(ob.m):
    axes[1].plot(sol2.t, sol2.y[i,:], alpha=1, lw=2.5)
axes[0].set_xlim([1,1.01])
axes[0].set_xlabel("Time")
axes[0].set_title("$n=3$ and $R_d=2000$")
axes[1].set_xlim([1,1.01])
axes[1].set_xlabel("Time")
axes[1].set_title("$n=3$ and $R_d=3000$")
plt.tight_layout()
plt.show() 