import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp 
from models.network import VSM
from models.explicit import ExplicitVSM
from models.ideal import LCLRsinglephase, LCLRsinglephase_localload

def generate_random_array(mu: float, nu: float) -> np.array:
    """
    Function to generate an array of length 10 with random numbers
    from a normal distribution with mean mu and variance nu.
    """
    random_array = np.random.normal(mu, np.sqrt(nu), 6)
    random_array[random_array < 0] = 0
    return random_array


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


def is_stable_explicit(model: object,
              n: int, 
              m: int, 
              Rload: float, 
              Rdlvec: np.array
              ) -> float:
    """
    Function to asses whether the explicit model is stable at 
    Rload with Rdlvec.
    """

    # Set up model
    ob = model() 
    ob.Rload = Rload
    x0 = np.zeros(n*m)
    ob.Rdlvec = Rdlvec
    ob.calculate_ref_powers() 

    #Timestep
    sol = solve_ivp(ob.F6_localloads, [0,1], x0, rtol=1e-10, method="LSODA", min_step=10e-8)
    # plt.plot(sol.y[1])
    # plt.show()

    # Check if steady state found
    if sol.status == 0: 
        return True 
    else: 
        return False



# # Plot the power demand of of the ideal lclc as a function of R
# inverter = LCLRsinglephase() 
# inverter.Ld = 1e-3
# R_transmission = 0.06
# inverter.R = 50.0 + R_transmission
# standard = 3*inverter.real_power()
# print(standard)

# # Plot the power demand of the ideal local load case with a very high
# # Rdl as a function of R
# inverter = LCLRsinglephase_localload() 
# inverter.Rdl = 10
# inverter.Lt = 1e-3
# R_transmission = 0.06
# powers_local = []
# Rdls = np.linspace(1, 1000000, 500)
# inverter.R = 50.0 + R_transmission
# for R in Rdls:
#     inverter.Rdl = R
#     powers_local.append(3*inverter.real_power())

# plt.semilogx(Rdls, powers_local, "m-")
# plt.axhline(standard)
# plt.show()


# # Find the Rcrit for n = 6
# crit_R = find_critical_point(VSM, 1.0)/6.0
# print(f"Crit T = {crit_R}")
crit_R = 1450.0

# Find the corresponding P value
inverter = LCLRsinglephase() 
inverter.Ld = 1e-3
R_transmission = 0.06
inverter.R = crit_R + R_transmission
bifurcation_point = 3*inverter.real_power()
print(f"Bif at {bifurcation_point}")


# print(Rdl_vec)
# print(np.mean(Rdl_vec))

# Plot P(mean Rdl) vs. variance
# 40 points, so 4 mus with 10 variances
mus = [30,40,45,50,55,60,65,70,75,80,85,90]
nus = [0,2,4,6,8,10,12,14,16,18,20,22,24,26]
# mus = [1e10]
# nus = [0.00]
powers = []
vars = []
stability = []
grid_load = 1400.0 #963.299560546875
for mu in mus:
    for nu in nus:

        # Create vector of Rdls
        Rdl_vec = generate_random_array(mu, nu)
        Rdl_mean = np.mean(Rdl_vec)

        # Calculate the power 
        inverter = LCLRsinglephase_localload() 
        inverter.Rdl = Rdl_mean
        inverter.Lt = 1e-3
        R_transmission = 0.06
        powers_local = []
        inverter.R = grid_load + R_transmission
        powers.append(3*inverter.real_power())
        vars.append(nu)

        # Assess stability
        print(f"Timestepping for mu = {mu}, nu = {nu}.")
        stable = is_stable_explicit(ExplicitVSM, 6, 13, grid_load, Rdl_vec)
        # stable = True
        print(f"Is it stable? {stable}")
        stability.append(stable)

# Plot resulting scatter chart
for i, power in enumerate(powers):
    if stability[i]:
        plt.plot(power, vars[i], "go")
    else:
        plt.plot(power, vars[i], "bo")
    
plt.axvline(bifurcation_point)
plt.show()
        

