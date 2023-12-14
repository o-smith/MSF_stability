import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp 
from models.network import VSM
from models.explicit import ExplicitVSM

# # Set up model
# ob = ExplicitVSM() 
# ob.n = 1
# ob.Rload =  50.0
# x0 = np.zeros(ob.n*ob.m*4 + 2)
# delta_R = 1000.0
# min_delta_R = 0.1
# ob.calculate_ref_powers()
# # ob.initialise_system() 
# # print(ob.Laplacian)

# # Bisect to find unstable point
# previously_stable = True
# while abs(delta_R) > min_delta_R:

#     print(f"Timestepping for R_load = {ob.Rload}")

#     #Time-step
#     # ob.initialise_system() 
#     ob.calculate_ref_powers()
#     sol = solve_ivp(ob.F4_comb, [0,1], x0, rtol=1e-10, method="LSODA", min_step=10e-8)

#     # Check if steady state found
#     if sol.status == 0: 
#         stable = True 
#     else: 
#         stable = False
    
#     # If there has been a change in stability
#     if stable != previously_stable:
#         delta_R = -delta_R/2.0
#         print(f"New delta = {delta_R}")

#     # Update
#     previously_stable = stable
#     ob.Rload += delta_R

# print(f"Critical R_load found at: {ob.Rload}")

# Your three numbers
## For tree: R_c = 1445.2
## For loop: R_c = 1533.2
## For comb: R_c = 2031.6
numbers = [1445.2, 1533.2, 2031.6]

# Positions for bars on the y-axis
positions = [1, 2, 3]

# Create a horizontal bar chart
plt.barh(positions, numbers)

# Add labels and title
# plt.xlabel('Values')
# plt.ylabel('Bars')
# plt.title('Horizontal Bar Chart')
plt.xticks([0,2100])
plt.xlim([0,2100])

# Set y-axis ticks to show the bar positions
plt.yticks(positions, ['Bar 1', 'Bar 2', 'Bar 3'])
plt.show()
# # Set up model 
# ob = ExplicitVSM()
# ob.Rload = 5750.0
# ob.n = 1
# x0 = np.zeros(ob.n*ob.m)
# delta_R = 1000.0
# min_delta_R = 0.1

# #Time-step
# ob.calculate_ref_powers()
# # ob.initialise_system()
# sol = solve_ivp(ob.F4, [0,5], x0, rtol=1e-10, method="LSODA", min_step=10e-8)
# for i in range(np.shape(sol.y[:,-1])[0]):
#     plt.plot(sol.t, sol.y[i,:], label=f"Param {i}")
# # plt.plot(sol.t, sol.y[3,:])
# # plt.legend()
# plt.show()

# # Bisect to find unstable point
# previously_stable = True
# while abs(delta_R) > min_delta_R:

#     print(f"Timestepping for R_load = {ob.Rload}")

#     #Time-step
#     ob.calculate_ref_powers()
#     sol = solve_ivp(ob.F4, [0,1], x0, rtol=1e-10, method="LSODA", min_step=10e-8)

#     # Check if steady state found
#     if sol.status == 0: 
#         stable = True 
#     else: 
#         stable = False
    
#     # If there has been a change in stability
#     if stable != previously_stable:
#         delta_R = -delta_R/2.0
#         print(f"New delta = {delta_R}")

#     # Update
#     previously_stable = stable
#     ob.Rload += delta_R

# print(f"Critical R_load found at: {ob.Rload}")

