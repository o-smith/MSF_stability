"""
While models/network.py containts matrix-vector formualtions
in the MSF style for control systems, this file contains 
equivalent implementations where ODEs are written out
explicitly. This has been used for testing the matrix-vector
formulations and for sanity testing time-simulations.
"""

import numpy as np

def split_array(input_array, split_num=4):
    """
    Function to take a numpy array of length (n*split_num) and return
    split_num individual arrays each of length n. This function is useful
    for breaking down the state vector for a large system during time-stepping.
    """
    if len(input_array) % split_num != 0:
        raise ValueError(f"Input array length must be divisible by {split_num}")
        
    # Split the transposed array into split_num arrays
    return np.split(input_array, split_num)
    


class ExplicitVSM:
    """
    Class to store explicit ODE models of VSM networks.
    """
    def __init__(self):
        self.record_count = 0 
        self.omegag = 2*np.pi*50 
        self.omega_ref = 2*np.pi*50 
        self.time = [] 
        self.P = [] 
        self.Q = [] 
        self.combinedP = []
        self.combinedVD = [] 
        self.combinedVQ = []
        self.va = [] 
        self.vb = [] 
        self.vc = [] 
        self.freq = [] 
        self.Uref = 400/np.sqrt(3) 
        self.L = 39e-3
        self.C = 2.2e-6
        self.Rt = 0.06
        self.Lt = 1e-3
        self.Rload = 60.0 
        self.n = 1 
        self.m = 13 
        self.Dp = 50
        self.J = 1
        self.K = 0.25
        self.Dq = 100
        self.rv = 0.1
        self.Kpv = 0.5
        self.Kpc = 10
        self.Kiv = 40
        self.Kic = 1500
        self.Rinf = 5e3


    def calculate_ref_powers(self, **kwargs):
        """
        Function to calculate reference power values.
        """
        self.__dict__.update(kwargs)

        # Calculate effective resistance
        Reff = self.Rload + self.Rt

        # Calculate phasor
        self.re_z = 1 - self.L*self.C*self.omega_ref**2 + (self.L*self.Lt*self.omega_ref**2)/(Reff**2 + self.omega_ref**2*self.Lt**2)
        self.im_z = (self.omega_ref*self.L*Reff)/(Reff**2 + self.omega_ref**2*self.Lt**2)

        # Calculate reference powers for each node
        self.Pref = 3*(self.Uref**2*Reff)/(2*(Reff**2+self.omega_ref**2*self.Lt**2))*(1/(self.re_z**2+self.im_z**2))
        self.Qref = 3*(self.Uref**2*self.omega_ref*self.Lt)/(2*(Reff**2+self.omega_ref**2*self.Lt**2))*(1/(self.re_z**2+self.im_z**2))


    def calculate_ref_powers_local_load(self, Rdelvec, **kwargs):
        """
        Function to compute the power for the case where there are
        additional local loads arranged in parallel.
        """
        self.__dict__.update(kwargs)

        self.Rdlvec = Rdelvec
        self.Rdl = np.mean(self.Rdlvec)

        # Calculate the effective resistance
        Rc = self.Rload + self.Rt
        Reff = self.Rdl*Rc/(self.Rdl + Rc)
        print(Reff)

        # Calculate phasor
        self.re_z = 1 - self.L*self.C*self.omega_ref**2 + (self.L*self.Lt*self.omega_ref**2)/(Reff**2 + self.omega_ref**2*self.Lt**2)
        self.im_z = (self.omega_ref*self.L*Reff)/(Reff**2 + self.omega_ref**2*self.Lt**2)

        # Calculate reference powers for each node
        self.Pref = 3*(self.Uref**2*Reff)/(2*(Reff**2+self.omega_ref**2*self.Lt**2))*(1/(self.re_z**2+self.im_z**2))
        self.Qref = 3*(self.Uref**2*self.omega_ref*self.Lt)/(2*(Reff**2+self.omega_ref**2*self.Lt**2))*(1/(self.re_z**2+self.im_z**2))


    def F4(self, _, x):
        """
        Function to return the RHS for the ODE system
        for 4 node VSM network in a star configuration.
        """
        # First, split the state vector x into the four 
        # state vectors for each node
        x1, x2, x3, x4 = split_array(x)

        # Now split each into the individual variables (longwinded!)
        i1_0d, i1_0q, v1_d, v1_q, i1_d, i1_q, omega1, phi1, e1, eps1_d, eps1_q, gam1_d, gam1_q = x1
        i2_0d, i2_0q, v2_d, v2_q, i2_d, i2_q, omega2, phi2, e2, eps2_d, eps2_q, gam2_d, gam2_q = x2
        i3_0d, i3_0q, v3_d, v3_q, i3_d, i3_q, omega3, phi3, e3, eps3_d, eps3_q, gam3_d, gam3_q = x3
        i4_0d, i4_0q, v4_d, v4_q, i4_d, i4_q, omega4, phi4, e4, eps4_d, eps4_q, gam4_d, gam4_q = x4

        # Now compute the control variables for each node
        # Starting with the control current
        i1_0dast = self.Kpv*(e1*np.cos(phi1) - self.rv*i1_d + omega1*self.L*i1_q - v1_d) + self.Kiv*eps1_d - omega1*self.C*v1_q
        i1_0qast = self.Kpv*(e1*np.sin(phi1) - self.rv*i1_q - omega1*self.L*i1_d - v1_q) + self.Kiv*eps1_q + omega1*self.C*v1_d
        i2_0dast = self.Kpv*(e2*np.cos(phi2) - self.rv*i2_d + omega2*self.L*i2_q - v2_d) + self.Kiv*eps2_d - omega2*self.C*v2_q
        i2_0qast = self.Kpv*(e2*np.sin(phi2) - self.rv*i2_q - omega2*self.L*i2_d - v2_q) + self.Kiv*eps2_q + omega2*self.C*v2_d
        i3_0dast = self.Kpv*(e3*np.cos(phi3) - self.rv*i3_d + omega3*self.L*i3_q - v3_d) + self.Kiv*eps3_d - omega3*self.C*v3_q
        i3_0qast = self.Kpv*(e3*np.sin(phi3) - self.rv*i3_q - omega3*self.L*i3_d - v3_q) + self.Kiv*eps3_q + omega3*self.C*v3_d
        i4_0dast = self.Kpv*(e4*np.cos(phi4) - self.rv*i4_d + omega4*self.L*i4_q - v4_d) + self.Kiv*eps4_d - omega4*self.C*v4_q
        i4_0qast = self.Kpv*(e4*np.sin(phi4) - self.rv*i4_q - omega4*self.L*i4_d - v4_q) + self.Kiv*eps4_q + omega4*self.C*v4_d

        # And now the input voltage for each node
        v1_0d = self.Kpc*(i1_0dast - i1_0d) + self.Kic*gam1_d - omega1*self.L*i1_0q
        v1_0q = self.Kpc*(i1_0qast - i1_0q) + self.Kic*gam1_q + omega1*self.L*i1_0d
        v2_0d = self.Kpc*(i2_0dast - i2_0d) + self.Kic*gam2_d - omega2*self.L*i2_0q
        v2_0q = self.Kpc*(i2_0qast - i2_0q) + self.Kic*gam2_q + omega2*self.L*i2_0d
        v3_0d = self.Kpc*(i3_0dast - i3_0d) + self.Kic*gam3_d - omega3*self.L*i3_0q
        v3_0q = self.Kpc*(i3_0qast - i3_0q) + self.Kic*gam3_q + omega3*self.L*i3_0d
        v4_0d = self.Kpc*(i4_0dast - i4_0d) + self.Kic*gam4_d - omega4*self.L*i4_0q
        v4_0q = self.Kpc*(i4_0qast - i4_0q) + self.Kic*gam4_q + omega4*self.L*i4_0d

        # Now the ODEs for each of the 4 nodes. Starting with node 1
        dx1 = np.zeros(len(x1))
        dx1[0] = (1.0/self.L)*(v1_0d - v1_d) + self.omegag*i1_0q
        dx1[1] = (1.0/self.L)*(v1_0q - v1_q) - self.omegag*i1_0d
        dx1[2] = (1.0/self.C)*(i1_0d - i1_d) + self.omegag*v1_q
        dx1[3] = (1.0/self.C)*(i1_0q - i1_q) - self.omegag*v1_d
        dx1[4] = (1.0/self.Lt)*(v1_d - i1_d*self.Rt) + self.omegag*i1_q - (self.Rload/self.Lt)*(i1_d + i2_d + i3_d + i4_d)
        dx1[5] = (1.0/self.Lt)*(v1_q - i1_q*self.Rt) - self.omegag*i1_d - (self.Rload/self.Lt)*(i1_q + i2_q + i3_q + i4_q)
        dx1[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v1_d*i1_d + v1_q*i1_q)) - (self.Dp*self.J)*(omega1 - self.omegag)
        dx1[7] = omega1 - self.omegag
        dx1[8] = (1/self.K)*(self.Qref - (3/2)*(v1_q*i1_d - v1_d*i1_q) - self.Dq*(np.sqrt((v1_d)**2 + (v1_q)**2) - self.Uref))
        dx1[9] = e1*np.cos(phi1) - self.rv*i1_d + omega1*self.L*i1_q - v1_d
        dx1[10] = e1*np.sin(phi1) - self.rv*i1_q - omega1*self.L*i1_d - v1_q
        dx1[11] = self.Kpv*(e1*np.cos(phi1) - self.rv*i1_d + omega1*self.L*i1_q - v1_d) + self.Kiv*eps1_d - omega1*self.C*v1_q - i1_0d
        dx1[12] = self.Kpv*(e1*np.sin(phi1) - self.rv*i1_q - omega1*self.L*i1_d - v1_q) + self.Kiv*eps1_q + omega1*self.C*v1_d - i1_0q

        # Now repeat for node 2
        dx2 = np.zeros(len(x2))
        dx2[0] = (1.0/self.L)*(v2_0d - v2_d) + self.omegag*i2_0q
        dx2[1] = (1.0/self.L)*(v2_0q - v2_q) - self.omegag*i2_0d
        dx2[2] = (1.0/self.C)*(i2_0d - i2_d) + self.omegag*v2_q
        dx2[3] = (1.0/self.C)*(i2_0q - i2_q) - self.omegag*v2_d
        dx2[4] = (1.0/self.Lt)*(v2_d - i2_d*self.Rt) + self.omegag*i2_q - (self.Rload/self.Lt)*(i1_d + i2_d + i3_d + i4_d)
        dx2[5] = (1.0/self.Lt)*(v2_q - i2_q*self.Rt) - self.omegag*i2_d - (self.Rload/self.Lt)*(i1_q + i2_q + i3_q + i4_q)
        dx2[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v2_d*i2_d + v2_q*i2_q)) - (self.Dp*self.J)*(omega2 - self.omegag)
        dx2[7] = omega2 - self.omegag
        dx2[8] = (1/self.K)*(self.Qref - (3/2)*(v2_q*i2_d - v2_d*i2_q) - self.Dq*(np.sqrt((v2_d)**2 + (v2_q)**2) - self.Uref))
        dx2[9] = e2*np.cos(phi2) - self.rv*i2_d + omega2*self.L*i2_q - v2_d
        dx2[10] = e2*np.sin(phi2) - self.rv*i2_q - omega2*self.L*i2_d - v2_q
        dx2[11] = self.Kpv*(e2*np.cos(phi2) - self.rv*i2_d + omega2*self.L*i2_q - v2_d) + self.Kiv*eps2_d - omega2*self.C*v2_q - i2_0d
        dx2[12] = self.Kpv*(e2*np.sin(phi2) - self.rv*i2_q - omega2*self.L*i2_d - v2_q) + self.Kiv*eps2_q + omega2*self.C*v2_d - i2_0q

        # Now repeat for node 3
        dx3 = np.zeros(len(x3))
        dx3[0] = (1.0/self.L)*(v3_0d - v3_d) + self.omegag*i3_0q
        dx3[1] = (1.0/self.L)*(v3_0q - v3_q) - self.omegag*i3_0d
        dx3[2] = (1.0/self.C)*(i3_0d - i3_d) + self.omegag*v3_q
        dx3[3] = (1.0/self.C)*(i3_0q - i3_q) - self.omegag*v3_d
        dx3[4] = (1.0/self.Lt)*(v3_d - i3_d*self.Rt) + self.omegag*i3_q - (self.Rload/self.Lt)*(i1_d + i2_d + i3_d + i4_d)
        dx3[5] = (1.0/self.Lt)*(v3_q - i3_q*self.Rt) - self.omegag*i3_d - (self.Rload/self.Lt)*(i1_q + i2_q + i3_q + i4_q)
        dx3[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v3_d*i3_d + v3_q*i3_q)) - (self.Dp*self.J)*(omega3 - self.omegag)
        dx3[7] = omega3 - self.omegag
        dx3[8] = (1/self.K)*(self.Qref - (3/2)*(v3_q*i3_d - v3_d*i3_q) - self.Dq*(np.sqrt((v3_d)**2 + (v3_q)**2) - self.Uref))
        dx3[9] = e3*np.cos(phi3) - self.rv*i3_d + omega3*self.L*i3_q - v3_d
        dx3[10] = e3*np.sin(phi3) - self.rv*i3_q - omega3*self.L*i3_d - v3_q
        dx3[11] = self.Kpv*(e3*np.cos(phi3) - self.rv*i3_d + omega3*self.L*i3_q - v3_d) + self.Kiv*eps3_d - omega3*self.C*v3_q - i3_0d
        dx3[12] = self.Kpv*(e3*np.sin(phi3) - self.rv*i3_q - omega3*self.L*i3_d - v3_q) + self.Kiv*eps3_q + omega3*self.C*v3_d - i3_0q

        # Now repeat for node 4
        dx4 = np.zeros(len(x4))
        dx4[0] = (1.0/self.L)*(v4_0d - v4_d) + self.omegag*i4_0q
        dx4[1] = (1.0/self.L)*(v4_0q - v4_q) - self.omegag*i4_0d
        dx4[2] = (1.0/self.C)*(i4_0d - i4_d) + self.omegag*v4_q
        dx4[3] = (1.0/self.C)*(i4_0q - i4_q) - self.omegag*v4_d
        dx4[4] = (1.0/self.Lt)*(v4_d - i4_d*self.Rt) + self.omegag*i4_q - (self.Rload/self.Lt)*(i1_d + i2_d + i3_d + i4_d)
        dx4[5] = (1.0/self.Lt)*(v4_q - i4_q*self.Rt) - self.omegag*i4_d - (self.Rload/self.Lt)*(i1_q + i2_q + i3_q + i4_q)
        dx4[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v4_d*i4_d + v4_q*i4_q)) - (self.Dp*self.J)*(omega4 - self.omegag)
        dx4[7] = omega4 - self.omegag
        dx4[8] = (1/self.K)*(self.Qref - (3/2)*(v4_q*i4_d - v4_d*i4_q) - self.Dq*(np.sqrt((v4_d)**2 + (v4_q)**2) - self.Uref))
        dx4[9] = e4*np.cos(phi4) - self.rv*i4_d + omega4*self.L*i4_q - v4_d
        dx4[10] = e4*np.sin(phi4) - self.rv*i4_q - omega4*self.L*i4_d - v4_q
        dx4[11] = self.Kpv*(e4*np.cos(phi4) - self.rv*i4_d + omega4*self.L*i4_q - v4_d) + self.Kiv*eps4_d - omega4*self.C*v4_q - i4_0d
        dx4[12] = self.Kpv*(e4*np.sin(phi4) - self.rv*i4_q - omega4*self.L*i4_d - v4_q) + self.Kiv*eps4_q + omega4*self.C*v4_d - i4_0q

        # return dx1
        return  np.concatenate([dx1, dx2, dx3, dx4])


    def F4_looped(self, _, x):
        """
        Function to return the RHS for the ODE system
        for 4 node VSM network in a looped configuration.
        """
        # First, split the state vector x into the four 
        # state vectors for each node
        x1 = x[0:13]
        x2 = x[13:26]
        x3 = x[26:39]
        x4 = x[39:52]
        i_iid = x[52]
        i_iiq = x[53]
        i_iiid = x[54]
        i_iiiq = x[55]
        i_vd = x[56]
        i_vq = x[57]

        # Now split each into the individual variables (longwinded!)
        i1_0d, i1_0q, v1_d, v1_q, i1_d, i1_q, omega1, phi1, e1, eps1_d, eps1_q, gam1_d, gam1_q = x1
        i2_0d, i2_0q, v2_d, v2_q, i2_d, i2_q, omega2, phi2, e2, eps2_d, eps2_q, gam2_d, gam2_q = x2
        i3_0d, i3_0q, v3_d, v3_q, i3_d, i3_q, omega3, phi3, e3, eps3_d, eps3_q, gam3_d, gam3_q = x3
        i4_0d, i4_0q, v4_d, v4_q, i4_d, i4_q, omega4, phi4, e4, eps4_d, eps4_q, gam4_d, gam4_q = x4

        # Compute the PCC voltages
        vpccd = self.Rload*(i1_d + i2_d + i_iid + i_vd)
        vpccq = self.Rload*(i1_q + i2_q + i_iiq + i_vq)

        # Now compute the control variables for each node
        # Starting with the control current
        i1_0dast = self.Kpv*(e1*np.cos(phi1) - self.rv*i1_d + omega1*self.L*i1_q - v1_d) + self.Kiv*eps1_d - omega1*self.C*v1_q
        i1_0qast = self.Kpv*(e1*np.sin(phi1) - self.rv*i1_q - omega1*self.L*i1_d - v1_q) + self.Kiv*eps1_q + omega1*self.C*v1_d
        i2_0dast = self.Kpv*(e2*np.cos(phi2) - self.rv*i2_d + omega2*self.L*i2_q - v2_d) + self.Kiv*eps2_d - omega2*self.C*v2_q
        i2_0qast = self.Kpv*(e2*np.sin(phi2) - self.rv*i2_q - omega2*self.L*i2_d - v2_q) + self.Kiv*eps2_q + omega2*self.C*v2_d
        i3_0dast = self.Kpv*(e3*np.cos(phi3) - self.rv*i3_d + omega3*self.L*i3_q - v3_d) + self.Kiv*eps3_d - omega3*self.C*v3_q
        i3_0qast = self.Kpv*(e3*np.sin(phi3) - self.rv*i3_q - omega3*self.L*i3_d - v3_q) + self.Kiv*eps3_q + omega3*self.C*v3_d
        i4_0dast = self.Kpv*(e4*np.cos(phi4) - self.rv*i4_d + omega4*self.L*i4_q - v4_d) + self.Kiv*eps4_d - omega4*self.C*v4_q
        i4_0qast = self.Kpv*(e4*np.sin(phi4) - self.rv*i4_q - omega4*self.L*i4_d - v4_q) + self.Kiv*eps4_q + omega4*self.C*v4_d

        # And now the input voltage for each node
        v1_0d = self.Kpc*(i1_0dast - i1_0d) + self.Kic*gam1_d - omega1*self.L*i1_0q
        v1_0q = self.Kpc*(i1_0qast - i1_0q) + self.Kic*gam1_q + omega1*self.L*i1_0d
        v2_0d = self.Kpc*(i2_0dast - i2_0d) + self.Kic*gam2_d - omega2*self.L*i2_0q
        v2_0q = self.Kpc*(i2_0qast - i2_0q) + self.Kic*gam2_q + omega2*self.L*i2_0d
        v3_0d = self.Kpc*(i3_0dast - i3_0d) + self.Kic*gam3_d - omega3*self.L*i3_0q
        v3_0q = self.Kpc*(i3_0qast - i3_0q) + self.Kic*gam3_q + omega3*self.L*i3_0d
        v4_0d = self.Kpc*(i4_0dast - i4_0d) + self.Kic*gam4_d - omega4*self.L*i4_0q
        v4_0q = self.Kpc*(i4_0qast - i4_0q) + self.Kic*gam4_q + omega4*self.L*i4_0d

        # Now the ODEs for each of the 4 nodes. Starting with node 1
        dx1 = np.zeros(len(x1))
        dx1[0] = (1.0/self.L)*(v1_0d - v1_d) + self.omegag*i1_0q
        dx1[1] = (1.0/self.L)*(v1_0q - v1_q) - self.omegag*i1_0d
        dx1[2] = (1.0/self.C)*(i1_0d - i1_d) + self.omegag*v1_q
        dx1[3] = (1.0/self.C)*(i1_0q - i1_q) - self.omegag*v1_d
        dx1[4] = (1.0/self.Lt)*(v1_d - i1_d*self.Rt) + self.omegag*i1_q - (vpccd/self.Lt)
        dx1[5] = (1.0/self.Lt)*(v1_q - i1_q*self.Rt) - self.omegag*i1_d - (vpccq/self.Lt)
        dx1[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v1_d*i1_d + v1_q*i1_q)) - (self.Dp*self.J)*(omega1 - self.omegag)
        dx1[7] = omega1 - self.omegag
        dx1[8] = (1/self.K)*(self.Qref - (3/2)*(v1_q*i1_d - v1_d*i1_q) - self.Dq*(np.sqrt((v1_d)**2 + (v1_q)**2) - self.Uref))
        dx1[9] = e1*np.cos(phi1) - self.rv*i1_d + omega1*self.L*i1_q - v1_d
        dx1[10] = e1*np.sin(phi1) - self.rv*i1_q - omega1*self.L*i1_d - v1_q
        dx1[11] = self.Kpv*(e1*np.cos(phi1) - self.rv*i1_d + omega1*self.L*i1_q - v1_d) + self.Kiv*eps1_d - omega1*self.C*v1_q - i1_0d
        dx1[12] = self.Kpv*(e1*np.sin(phi1) - self.rv*i1_q - omega1*self.L*i1_d - v1_q) + self.Kiv*eps1_q + omega1*self.C*v1_d - i1_0q

        # Now repeat for node 2
        dx2 = np.zeros(len(x2))
        dx2[0] = (1.0/self.L)*(v2_0d - v2_d) + self.omegag*i2_0q
        dx2[1] = (1.0/self.L)*(v2_0q - v2_q) - self.omegag*i2_0d
        dx2[2] = (1.0/self.C)*(i2_0d - i2_d) + self.omegag*v2_q
        dx2[3] = (1.0/self.C)*(i2_0q - i2_q) - self.omegag*v2_d
        dx2[4] = (1.0/self.Lt)*(v2_d - i2_d*self.Rt) + self.omegag*i2_q - (vpccd/self.Lt)
        dx2[5] = (1.0/self.Lt)*(v2_q - i2_q*self.Rt) - self.omegag*i2_d - (vpccq/self.Lt)
        dx2[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v2_d*i2_d + v2_q*i2_q)) - (self.Dp*self.J)*(omega2 - self.omegag)
        dx2[7] = omega2 - self.omegag
        dx2[8] = (1/self.K)*(self.Qref - (3/2)*(v2_q*i2_d - v2_d*i2_q) - self.Dq*(np.sqrt((v2_d)**2 + (v2_q)**2) - self.Uref))
        dx2[9] = e2*np.cos(phi2) - self.rv*i2_d + omega2*self.L*i2_q - v2_d
        dx2[10] = e2*np.sin(phi2) - self.rv*i2_q - omega2*self.L*i2_d - v2_q
        dx2[11] = self.Kpv*(e2*np.cos(phi2) - self.rv*i2_d + omega2*self.L*i2_q - v2_d) + self.Kiv*eps2_d - omega2*self.C*v2_q - i2_0d
        dx2[12] = self.Kpv*(e2*np.sin(phi2) - self.rv*i2_q - omega2*self.L*i2_d - v2_q) + self.Kiv*eps2_q + omega2*self.C*v2_d - i2_0q

        # Now repeat for node 3
        dx3 = np.zeros(len(x3))
        dx3[0] = (1.0/self.L)*(v3_0d - v3_d) + self.omegag*i3_0q
        dx3[1] = (1.0/self.L)*(v3_0q - v3_q) - self.omegag*i3_0d
        dx3[2] = (1.0/self.C)*(i3_0d - i3_d) + self.omegag*v3_q
        dx3[3] = (1.0/self.C)*(i3_0q - i3_q) - self.omegag*v3_d
        dx3[4] = (1.0/self.Lt)*(v3_d - i3_d*self.Rt) + self.omegag*i3_q - (self.Rinf/self.Lt)*(i3_d - i_iid - i_iiid)
        dx3[5] = (1.0/self.Lt)*(v3_q - i3_q*self.Rt) - self.omegag*i3_d - (self.Rinf/self.Lt)*(i3_q - i_iiq - i_iiiq)
        dx3[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v3_d*i3_d + v3_q*i3_q)) - (self.Dp*self.J)*(omega3 - self.omegag)
        dx3[7] = omega3 - self.omegag
        dx3[8] = (1/self.K)*(self.Qref - (3/2)*(v3_q*i3_d - v3_d*i3_q) - self.Dq*(np.sqrt((v3_d)**2 + (v3_q)**2) - self.Uref))
        dx3[9] = e3*np.cos(phi3) - self.rv*i3_d + omega3*self.L*i3_q - v3_d
        dx3[10] = e3*np.sin(phi3) - self.rv*i3_q - omega3*self.L*i3_d - v3_q
        dx3[11] = self.Kpv*(e3*np.cos(phi3) - self.rv*i3_d + omega3*self.L*i3_q - v3_d) + self.Kiv*eps3_d - omega3*self.C*v3_q - i3_0d
        dx3[12] = self.Kpv*(e3*np.sin(phi3) - self.rv*i3_q - omega3*self.L*i3_d - v3_q) + self.Kiv*eps3_q + omega3*self.C*v3_d - i3_0q

        # Now repeat for node 4
        dx4 = np.zeros(len(x4))
        dx4[0] = (1.0/self.L)*(v4_0d - v4_d) + self.omegag*i4_0q
        dx4[1] = (1.0/self.L)*(v4_0q - v4_q) - self.omegag*i4_0d
        dx4[2] = (1.0/self.C)*(i4_0d - i4_d) + self.omegag*v4_q
        dx4[3] = (1.0/self.C)*(i4_0q - i4_q) - self.omegag*v4_d
        dx4[4] = (1.0/self.Lt)*(v4_d - i4_d*self.Rt) + self.omegag*i4_q - (self.Rinf/self.Lt)*(i4_d + i_iiid - i_vd)
        dx4[5] = (1.0/self.Lt)*(v4_q - i4_q*self.Rt) - self.omegag*i4_d - (self.Rinf/self.Lt)*(i4_q + i_iiiq - i_vq)
        dx4[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v4_d*i4_d + v4_q*i4_q)) - (self.Dp*self.J)*(omega4 - self.omegag)
        dx4[7] = omega4 - self.omegag
        dx4[8] = (1/self.K)*(self.Qref - (3/2)*(v4_q*i4_d - v4_d*i4_q) - self.Dq*(np.sqrt((v4_d)**2 + (v4_q)**2) - self.Uref))
        dx4[9] = e4*np.cos(phi4) - self.rv*i4_d + omega4*self.L*i4_q - v4_d
        dx4[10] = e4*np.sin(phi4) - self.rv*i4_q - omega4*self.L*i4_d - v4_q
        dx4[11] = self.Kpv*(e4*np.cos(phi4) - self.rv*i4_d + omega4*self.L*i4_q - v4_d) + self.Kiv*eps4_d - omega4*self.C*v4_q - i4_0d
        dx4[12] = self.Kpv*(e4*np.sin(phi4) - self.rv*i4_q - omega4*self.L*i4_d - v4_q) + self.Kiv*eps4_q + omega4*self.C*v4_d - i4_0q

        # Extra 6 equations for the loop
        di_iid = (self.Rinf/self.Lt)*(i3_d - i_iid - i_iiid) - (vpccd/self.Lt) - (self.Rt/self.Lt)*i_iid + self.omegag*i_iiq
        di_iiq = (self.Rinf/self.Lt)*(i3_q - i_iiq - i_iiiq) - (vpccq/self.Lt) - (self.Rt/self.Lt)*i_iiq - self.omegag*i_iid
        di_iiid = (self.Rinf/self.Lt)*(i3_d - i_iid - i_iiid - i4_d - i_iiid + i_vd) - (self.Rt/self.Lt)*i_iiid + self.omegag*i_iiiq
        di_iiiq = (self.Rinf/self.Lt)*(i3_q - i_iiq - i_iiiq - i4_q - i_iiiq + i_vq) - (self.Rt/self.Lt)*i_iiiq - self.omegag*i_iiid
        di_vd = (self.Rinf/self.Lt)*(i4_d + i_iiid - i_vd) - (vpccd/self.Lt) - (self.Rt/self.Lt)*i_vd + self.omegag*i_vq
        di_vq = (self.Rinf/self.Lt)*(i4_q + i_iiiq - i_vq) - (vpccq/self.Lt) - (self.Rt/self.Lt)*i_vq - self.omegag*i_vd

        # return dx1
        return np.concatenate([dx1, dx2, dx3, dx4, [di_iid],[di_iiq],[di_iiid],[di_iiiq],[di_vd],[di_vq]])


    def F4_comb(self, _, x):
        """
        Function to return the RHS for the ODE system
        for 4 node VSM network in an asymmetric branched configuration.
        """
        # First, split the state vector x into the four 
        # state vectors for each node
        x1 = x[0:13]
        x2 = x[13:26]
        x3 = x[26:39]
        x4 = x[39:52]
        i_combd = x[52]
        i_combq = x[53]

        # Now split each into the individual variables (longwinded!)
        i1_0d, i1_0q, v1_d, v1_q, i1_d, i1_q, omega1, phi1, e1, eps1_d, eps1_q, gam1_d, gam1_q = x1
        i2_0d, i2_0q, v2_d, v2_q, i2_d, i2_q, omega2, phi2, e2, eps2_d, eps2_q, gam2_d, gam2_q = x2
        i3_0d, i3_0q, v3_d, v3_q, i3_d, i3_q, omega3, phi3, e3, eps3_d, eps3_q, gam3_d, gam3_q = x3
        i4_0d, i4_0q, v4_d, v4_q, i4_d, i4_q, omega4, phi4, e4, eps4_d, eps4_q, gam4_d, gam4_q = x4

        # Compute the PCC voltages
        vpccd = self.Rload*(i1_d + i2_d + i_combd)
        vpccq = self.Rload*(i1_q + i2_q + i_combq)

        # Now compute the control variables for each node
        # Starting with the control current
        i1_0dast = self.Kpv*(e1*np.cos(phi1) - self.rv*i1_d + omega1*self.L*i1_q - v1_d) + self.Kiv*eps1_d - omega1*self.C*v1_q
        i1_0qast = self.Kpv*(e1*np.sin(phi1) - self.rv*i1_q - omega1*self.L*i1_d - v1_q) + self.Kiv*eps1_q + omega1*self.C*v1_d
        i2_0dast = self.Kpv*(e2*np.cos(phi2) - self.rv*i2_d + omega2*self.L*i2_q - v2_d) + self.Kiv*eps2_d - omega2*self.C*v2_q
        i2_0qast = self.Kpv*(e2*np.sin(phi2) - self.rv*i2_q - omega2*self.L*i2_d - v2_q) + self.Kiv*eps2_q + omega2*self.C*v2_d
        i3_0dast = self.Kpv*(e3*np.cos(phi3) - self.rv*i3_d + omega3*self.L*i3_q - v3_d) + self.Kiv*eps3_d - omega3*self.C*v3_q
        i3_0qast = self.Kpv*(e3*np.sin(phi3) - self.rv*i3_q - omega3*self.L*i3_d - v3_q) + self.Kiv*eps3_q + omega3*self.C*v3_d
        i4_0dast = self.Kpv*(e4*np.cos(phi4) - self.rv*i4_d + omega4*self.L*i4_q - v4_d) + self.Kiv*eps4_d - omega4*self.C*v4_q
        i4_0qast = self.Kpv*(e4*np.sin(phi4) - self.rv*i4_q - omega4*self.L*i4_d - v4_q) + self.Kiv*eps4_q + omega4*self.C*v4_d

        # And now the input voltage for each node
        v1_0d = self.Kpc*(i1_0dast - i1_0d) + self.Kic*gam1_d - omega1*self.L*i1_0q
        v1_0q = self.Kpc*(i1_0qast - i1_0q) + self.Kic*gam1_q + omega1*self.L*i1_0d
        v2_0d = self.Kpc*(i2_0dast - i2_0d) + self.Kic*gam2_d - omega2*self.L*i2_0q
        v2_0q = self.Kpc*(i2_0qast - i2_0q) + self.Kic*gam2_q + omega2*self.L*i2_0d
        v3_0d = self.Kpc*(i3_0dast - i3_0d) + self.Kic*gam3_d - omega3*self.L*i3_0q
        v3_0q = self.Kpc*(i3_0qast - i3_0q) + self.Kic*gam3_q + omega3*self.L*i3_0d
        v4_0d = self.Kpc*(i4_0dast - i4_0d) + self.Kic*gam4_d - omega4*self.L*i4_0q
        v4_0q = self.Kpc*(i4_0qast - i4_0q) + self.Kic*gam4_q + omega4*self.L*i4_0d

        # Now the ODEs for each of the 4 nodes. Starting with node 1
        dx1 = np.zeros(len(x1))
        dx1[0] = (1.0/self.L)*(v1_0d - v1_d) + self.omegag*i1_0q
        dx1[1] = (1.0/self.L)*(v1_0q - v1_q) - self.omegag*i1_0d
        dx1[2] = (1.0/self.C)*(i1_0d - i1_d) + self.omegag*v1_q
        dx1[3] = (1.0/self.C)*(i1_0q - i1_q) - self.omegag*v1_d
        dx1[4] = (1.0/self.Lt)*(v1_d - i1_d*self.Rt) + self.omegag*i1_q - (vpccd/self.Lt)
        dx1[5] = (1.0/self.Lt)*(v1_q - i1_q*self.Rt) - self.omegag*i1_d - (vpccq/self.Lt)
        dx1[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v1_d*i1_d + v1_q*i1_q)) - (self.Dp*self.J)*(omega1 - self.omegag)
        dx1[7] = omega1 - self.omegag
        dx1[8] = (1/self.K)*(self.Qref - (3/2)*(v1_q*i1_d - v1_d*i1_q) - self.Dq*(np.sqrt((v1_d)**2 + (v1_q)**2) - self.Uref))
        dx1[9] = e1*np.cos(phi1) - self.rv*i1_d + omega1*self.L*i1_q - v1_d
        dx1[10] = e1*np.sin(phi1) - self.rv*i1_q - omega1*self.L*i1_d - v1_q
        dx1[11] = self.Kpv*(e1*np.cos(phi1) - self.rv*i1_d + omega1*self.L*i1_q - v1_d) + self.Kiv*eps1_d - omega1*self.C*v1_q - i1_0d
        dx1[12] = self.Kpv*(e1*np.sin(phi1) - self.rv*i1_q - omega1*self.L*i1_d - v1_q) + self.Kiv*eps1_q + omega1*self.C*v1_d - i1_0q

        # Now repeat for node 2
        dx2 = np.zeros(len(x2))
        dx2[0] = (1.0/self.L)*(v2_0d - v2_d) + self.omegag*i2_0q
        dx2[1] = (1.0/self.L)*(v2_0q - v2_q) - self.omegag*i2_0d
        dx2[2] = (1.0/self.C)*(i2_0d - i2_d) + self.omegag*v2_q
        dx2[3] = (1.0/self.C)*(i2_0q - i2_q) - self.omegag*v2_d
        dx2[4] = (1.0/self.Lt)*(v2_d - i2_d*self.Rt) + self.omegag*i2_q - (vpccd/self.Lt)
        dx2[5] = (1.0/self.Lt)*(v2_q - i2_q*self.Rt) - self.omegag*i2_d - (vpccq/self.Lt)
        dx2[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v2_d*i2_d + v2_q*i2_q)) - (self.Dp*self.J)*(omega2 - self.omegag)
        dx2[7] = omega2 - self.omegag
        dx2[8] = (1/self.K)*(self.Qref - (3/2)*(v2_q*i2_d - v2_d*i2_q) - self.Dq*(np.sqrt((v2_d)**2 + (v2_q)**2) - self.Uref))
        dx2[9] = e2*np.cos(phi2) - self.rv*i2_d + omega2*self.L*i2_q - v2_d
        dx2[10] = e2*np.sin(phi2) - self.rv*i2_q - omega2*self.L*i2_d - v2_q
        dx2[11] = self.Kpv*(e2*np.cos(phi2) - self.rv*i2_d + omega2*self.L*i2_q - v2_d) + self.Kiv*eps2_d - omega2*self.C*v2_q - i2_0d
        dx2[12] = self.Kpv*(e2*np.sin(phi2) - self.rv*i2_q - omega2*self.L*i2_d - v2_q) + self.Kiv*eps2_q + omega2*self.C*v2_d - i2_0q

        # Now repeat for node 3
        dx3 = np.zeros(len(x3))
        dx3[0] = (1.0/self.L)*(v3_0d - v3_d) + self.omegag*i3_0q
        dx3[1] = (1.0/self.L)*(v3_0q - v3_q) - self.omegag*i3_0d
        dx3[2] = (1.0/self.C)*(i3_0d - i3_d) + self.omegag*v3_q
        dx3[3] = (1.0/self.C)*(i3_0q - i3_q) - self.omegag*v3_d
        dx3[4] = (1.0/self.Lt)*(v3_d - i3_d*self.Rt) + self.omegag*i3_q - (self.Rinf/self.Lt)*(i3_d + i4_d - i_combd)
        dx3[5] = (1.0/self.Lt)*(v3_q - i3_q*self.Rt) - self.omegag*i3_d - (self.Rinf/self.Lt)*(i3_q + i4_q - i_combq)
        dx3[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v3_d*i3_d + v3_q*i3_q)) - (self.Dp*self.J)*(omega3 - self.omegag)
        dx3[7] = omega3 - self.omegag
        dx3[8] = (1/self.K)*(self.Qref - (3/2)*(v3_q*i3_d - v3_d*i3_q) - self.Dq*(np.sqrt((v3_d)**2 + (v3_q)**2) - self.Uref))
        dx3[9] = e3*np.cos(phi3) - self.rv*i3_d + omega3*self.L*i3_q - v3_d
        dx3[10] = e3*np.sin(phi3) - self.rv*i3_q - omega3*self.L*i3_d - v3_q
        dx3[11] = self.Kpv*(e3*np.cos(phi3) - self.rv*i3_d + omega3*self.L*i3_q - v3_d) + self.Kiv*eps3_d - omega3*self.C*v3_q - i3_0d
        dx3[12] = self.Kpv*(e3*np.sin(phi3) - self.rv*i3_q - omega3*self.L*i3_d - v3_q) + self.Kiv*eps3_q + omega3*self.C*v3_d - i3_0q

        # Now repeat for node 4
        dx4 = np.zeros(len(x4))
        dx4[0] = (1.0/self.L)*(v4_0d - v4_d) + self.omegag*i4_0q
        dx4[1] = (1.0/self.L)*(v4_0q - v4_q) - self.omegag*i4_0d
        dx4[2] = (1.0/self.C)*(i4_0d - i4_d) + self.omegag*v4_q
        dx4[3] = (1.0/self.C)*(i4_0q - i4_q) - self.omegag*v4_d
        dx4[4] = (1.0/self.Lt)*(v4_d - i4_d*self.Rt) + self.omegag*i4_q - (self.Rinf/self.Lt)*(i3_d + i4_d - i_combd)
        dx4[5] = (1.0/self.Lt)*(v4_q - i4_q*self.Rt) - self.omegag*i4_d - (self.Rinf/self.Lt)*(i3_q + i4_q - i_combq)
        dx4[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v4_d*i4_d + v4_q*i4_q)) - (self.Dp*self.J)*(omega4 - self.omegag)
        dx4[7] = omega4 - self.omegag
        dx4[8] = (1/self.K)*(self.Qref - (3/2)*(v4_q*i4_d - v4_d*i4_q) - self.Dq*(np.sqrt((v4_d)**2 + (v4_q)**2) - self.Uref))
        dx4[9] = e4*np.cos(phi4) - self.rv*i4_d + omega4*self.L*i4_q - v4_d
        dx4[10] = e4*np.sin(phi4) - self.rv*i4_q - omega4*self.L*i4_d - v4_q
        dx4[11] = self.Kpv*(e4*np.cos(phi4) - self.rv*i4_d + omega4*self.L*i4_q - v4_d) + self.Kiv*eps4_d - omega4*self.C*v4_q - i4_0d
        dx4[12] = self.Kpv*(e4*np.sin(phi4) - self.rv*i4_q - omega4*self.L*i4_d - v4_q) + self.Kiv*eps4_q + omega4*self.C*v4_d - i4_0q

        # Extra 2 equations for the combined branch
        di_combd = (self.Rinf/self.Lt)*(i3_d + i4_d - i_combd) - (self.Rt/self.Lt)*i_combd - (vpccd/self.Lt) + self.omegag*i_combq
        di_combq = (self.Rinf/self.Lt)*(i3_q + i4_q - i_combq) - (self.Rt/self.Lt)*i_combq - (vpccq/self.Lt) - self.omegag*i_combd

        # return dx1
        return np.concatenate([dx1, dx2, dx3, dx4, [di_combd],[di_combq]])
    

    def F6_localloads(self, _, x):
        """
        Function to return the RHS for the ODE system
        for 4 node VSM network in a star configuration
        with heterogeneous local loads.
        """
        # First, split the state vector x into the four 
        # state vectors for each node
        x1, x2, x3, x4, x5, x6 = split_array(x, split_num=6)

        # Now split each into the individual variables (longwinded!)
        i1_0d, i1_0q, v1_d, v1_q, i1_d, i1_q, omega1, phi1, e1, eps1_d, eps1_q, gam1_d, gam1_q = x1
        i2_0d, i2_0q, v2_d, v2_q, i2_d, i2_q, omega2, phi2, e2, eps2_d, eps2_q, gam2_d, gam2_q = x2
        i3_0d, i3_0q, v3_d, v3_q, i3_d, i3_q, omega3, phi3, e3, eps3_d, eps3_q, gam3_d, gam3_q = x3
        i4_0d, i4_0q, v4_d, v4_q, i4_d, i4_q, omega4, phi4, e4, eps4_d, eps4_q, gam4_d, gam4_q = x4
        i5_0d, i5_0q, v5_d, v5_q, i5_d, i5_q, omega5, phi5, e5, eps5_d, eps5_q, gam5_d, gam5_q = x5
        i6_0d, i6_0q, v6_d, v6_q, i6_d, i6_q, omega6, phi6, e6, eps6_d, eps6_q, gam6_d, gam6_q = x6

        # Now compute the control variables for each node
        # Starting with the control current
        i1_0dast = self.Kpv*(e1*np.cos(phi1) - self.rv*i1_d + omega1*self.L*i1_q - v1_d) + self.Kiv*eps1_d - omega1*self.C*v1_q
        i1_0qast = self.Kpv*(e1*np.sin(phi1) - self.rv*i1_q - omega1*self.L*i1_d - v1_q) + self.Kiv*eps1_q + omega1*self.C*v1_d
        i2_0dast = self.Kpv*(e2*np.cos(phi2) - self.rv*i2_d + omega2*self.L*i2_q - v2_d) + self.Kiv*eps2_d - omega2*self.C*v2_q
        i2_0qast = self.Kpv*(e2*np.sin(phi2) - self.rv*i2_q - omega2*self.L*i2_d - v2_q) + self.Kiv*eps2_q + omega2*self.C*v2_d
        i3_0dast = self.Kpv*(e3*np.cos(phi3) - self.rv*i3_d + omega3*self.L*i3_q - v3_d) + self.Kiv*eps3_d - omega3*self.C*v3_q
        i3_0qast = self.Kpv*(e3*np.sin(phi3) - self.rv*i3_q - omega3*self.L*i3_d - v3_q) + self.Kiv*eps3_q + omega3*self.C*v3_d
        i4_0dast = self.Kpv*(e4*np.cos(phi4) - self.rv*i4_d + omega4*self.L*i4_q - v4_d) + self.Kiv*eps4_d - omega4*self.C*v4_q
        i4_0qast = self.Kpv*(e4*np.sin(phi4) - self.rv*i4_q - omega4*self.L*i4_d - v4_q) + self.Kiv*eps4_q + omega4*self.C*v4_d
        i5_0dast = self.Kpv*(e5*np.cos(phi5) - self.rv*i5_d + omega5*self.L*i5_q - v5_d) + self.Kiv*eps5_d - omega5*self.C*v5_q
        i5_0qast = self.Kpv*(e5*np.sin(phi5) - self.rv*i5_q - omega5*self.L*i5_d - v5_q) + self.Kiv*eps5_q + omega5*self.C*v5_d
        i6_0dast = self.Kpv*(e6*np.cos(phi6) - self.rv*i6_d + omega6*self.L*i6_q - v6_d) + self.Kiv*eps6_d - omega6*self.C*v6_q
        i6_0qast = self.Kpv*(e6*np.sin(phi6) - self.rv*i6_q - omega6*self.L*i6_d - v6_q) + self.Kiv*eps6_q + omega6*self.C*v6_d

        # And now the input voltage for each node
        v1_0d = self.Kpc*(i1_0dast - i1_0d) + self.Kic*gam1_d - omega1*self.L*i1_0q
        v1_0q = self.Kpc*(i1_0qast - i1_0q) + self.Kic*gam1_q + omega1*self.L*i1_0d
        v2_0d = self.Kpc*(i2_0dast - i2_0d) + self.Kic*gam2_d - omega2*self.L*i2_0q
        v2_0q = self.Kpc*(i2_0qast - i2_0q) + self.Kic*gam2_q + omega2*self.L*i2_0d
        v3_0d = self.Kpc*(i3_0dast - i3_0d) + self.Kic*gam3_d - omega3*self.L*i3_0q
        v3_0q = self.Kpc*(i3_0qast - i3_0q) + self.Kic*gam3_q + omega3*self.L*i3_0d
        v4_0d = self.Kpc*(i4_0dast - i4_0d) + self.Kic*gam4_d - omega4*self.L*i4_0q
        v4_0q = self.Kpc*(i4_0qast - i4_0q) + self.Kic*gam4_q + omega4*self.L*i4_0d
        v5_0d = self.Kpc*(i5_0dast - i5_0d) + self.Kic*gam5_d - omega5*self.L*i5_0q
        v5_0q = self.Kpc*(i5_0qast - i5_0q) + self.Kic*gam5_q + omega5*self.L*i5_0d
        v6_0d = self.Kpc*(i6_0dast - i6_0d) + self.Kic*gam6_d - omega6*self.L*i6_0q
        v6_0q = self.Kpc*(i6_0qast - i6_0q) + self.Kic*gam6_q + omega6*self.L*i6_0d

        # Split the local load vector up
        Rdl1, Rdl2, Rdl3, Rdl4, Rdl5, Rdl6 = self.Rdlvec

        # Now the ODEs for each of the 4 nodes. Starting with node 1
        Rc1 = (self.Rload*Rdl1)/(self.Rload + Rdl1)
        dx1 = np.zeros(len(x1))
        dx1[0] = (1.0/self.L)*(v1_0d - v1_d) + self.omegag*i1_0q
        dx1[1] = (1.0/self.L)*(v1_0q - v1_q) - self.omegag*i1_0d
        dx1[2] = (1.0/self.C)*(i1_0d - i1_d) + self.omegag*v1_q
        dx1[3] = (1.0/self.C)*(i1_0q - i1_q) - self.omegag*v1_d
        dx1[4] = (1.0/self.Lt)*(v1_d - i1_d*self.Rt) + self.omegag*i1_q - (Rc1/self.Lt)*(i1_d + i2_d + i3_d + i4_d)
        dx1[5] = (1.0/self.Lt)*(v1_q - i1_q*self.Rt) - self.omegag*i1_d - (Rc1/self.Lt)*(i1_q + i2_q + i3_q + i4_q)
        dx1[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v1_d*i1_d + v1_q*i1_q)) - (self.Dp*self.J)*(omega1 - self.omegag)
        dx1[7] = omega1 - self.omegag
        dx1[8] = (1/self.K)*(self.Qref - (3/2)*(v1_q*i1_d - v1_d*i1_q) - self.Dq*(np.sqrt((v1_d)**2 + (v1_q)**2) - self.Uref))
        dx1[9] = e1*np.cos(phi1) - self.rv*i1_d + omega1*self.L*i1_q - v1_d
        dx1[10] = e1*np.sin(phi1) - self.rv*i1_q - omega1*self.L*i1_d - v1_q
        dx1[11] = self.Kpv*(e1*np.cos(phi1) - self.rv*i1_d + omega1*self.L*i1_q - v1_d) + self.Kiv*eps1_d - omega1*self.C*v1_q - i1_0d
        dx1[12] = self.Kpv*(e1*np.sin(phi1) - self.rv*i1_q - omega1*self.L*i1_d - v1_q) + self.Kiv*eps1_q + omega1*self.C*v1_d - i1_0q

        # Now repeat for node 2
        Rc2 = (self.Rload*Rdl2)/(self.Rload + Rdl2)
        dx2 = np.zeros(len(x2))
        dx2[0] = (1.0/self.L)*(v2_0d - v2_d) + self.omegag*i2_0q
        dx2[1] = (1.0/self.L)*(v2_0q - v2_q) - self.omegag*i2_0d
        dx2[2] = (1.0/self.C)*(i2_0d - i2_d) + self.omegag*v2_q
        dx2[3] = (1.0/self.C)*(i2_0q - i2_q) - self.omegag*v2_d
        dx2[4] = (1.0/self.Lt)*(v2_d - i2_d*self.Rt) + self.omegag*i2_q - (Rc2/self.Lt)*(i1_d + i2_d + i3_d + i4_d)
        dx2[5] = (1.0/self.Lt)*(v2_q - i2_q*self.Rt) - self.omegag*i2_d - (Rc2/self.Lt)*(i1_q + i2_q + i3_q + i4_q)
        dx2[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v2_d*i2_d + v2_q*i2_q)) - (self.Dp*self.J)*(omega2 - self.omegag)
        dx2[7] = omega2 - self.omegag
        dx2[8] = (1/self.K)*(self.Qref - (3/2)*(v2_q*i2_d - v2_d*i2_q) - self.Dq*(np.sqrt((v2_d)**2 + (v2_q)**2) - self.Uref))
        dx2[9] = e2*np.cos(phi2) - self.rv*i2_d + omega2*self.L*i2_q - v2_d
        dx2[10] = e2*np.sin(phi2) - self.rv*i2_q - omega2*self.L*i2_d - v2_q
        dx2[11] = self.Kpv*(e2*np.cos(phi2) - self.rv*i2_d + omega2*self.L*i2_q - v2_d) + self.Kiv*eps2_d - omega2*self.C*v2_q - i2_0d
        dx2[12] = self.Kpv*(e2*np.sin(phi2) - self.rv*i2_q - omega2*self.L*i2_d - v2_q) + self.Kiv*eps2_q + omega2*self.C*v2_d - i2_0q

        # Now repeat for node 3
        Rc3 = (self.Rload*Rdl3)/(self.Rload + Rdl3)
        dx3 = np.zeros(len(x3))
        dx3[0] = (1.0/self.L)*(v3_0d - v3_d) + self.omegag*i3_0q
        dx3[1] = (1.0/self.L)*(v3_0q - v3_q) - self.omegag*i3_0d
        dx3[2] = (1.0/self.C)*(i3_0d - i3_d) + self.omegag*v3_q
        dx3[3] = (1.0/self.C)*(i3_0q - i3_q) - self.omegag*v3_d
        dx3[4] = (1.0/self.Lt)*(v3_d - i3_d*self.Rt) + self.omegag*i3_q - (Rc3/self.Lt)*(i1_d + i2_d + i3_d + i4_d)
        dx3[5] = (1.0/self.Lt)*(v3_q - i3_q*self.Rt) - self.omegag*i3_d - (Rc3/self.Lt)*(i1_q + i2_q + i3_q + i4_q)
        dx3[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v3_d*i3_d + v3_q*i3_q)) - (self.Dp*self.J)*(omega3 - self.omegag)
        dx3[7] = omega3 - self.omegag
        dx3[8] = (1/self.K)*(self.Qref - (3/2)*(v3_q*i3_d - v3_d*i3_q) - self.Dq*(np.sqrt((v3_d)**2 + (v3_q)**2) - self.Uref))
        dx3[9] = e3*np.cos(phi3) - self.rv*i3_d + omega3*self.L*i3_q - v3_d
        dx3[10] = e3*np.sin(phi3) - self.rv*i3_q - omega3*self.L*i3_d - v3_q
        dx3[11] = self.Kpv*(e3*np.cos(phi3) - self.rv*i3_d + omega3*self.L*i3_q - v3_d) + self.Kiv*eps3_d - omega3*self.C*v3_q - i3_0d
        dx3[12] = self.Kpv*(e3*np.sin(phi3) - self.rv*i3_q - omega3*self.L*i3_d - v3_q) + self.Kiv*eps3_q + omega3*self.C*v3_d - i3_0q

        # Now repeat for node 4
        Rc4 = (self.Rload*Rdl4)/(self.Rload + Rdl4)
        dx4 = np.zeros(len(x4))
        dx4[0] = (1.0/self.L)*(v4_0d - v4_d) + self.omegag*i4_0q
        dx4[1] = (1.0/self.L)*(v4_0q - v4_q) - self.omegag*i4_0d
        dx4[2] = (1.0/self.C)*(i4_0d - i4_d) + self.omegag*v4_q
        dx4[3] = (1.0/self.C)*(i4_0q - i4_q) - self.omegag*v4_d
        dx4[4] = (1.0/self.Lt)*(v4_d - i4_d*self.Rt) + self.omegag*i4_q - (Rc4/self.Lt)*(i1_d + i2_d + i3_d + i4_d)
        dx4[5] = (1.0/self.Lt)*(v4_q - i4_q*self.Rt) - self.omegag*i4_d - (Rc4/self.Lt)*(i1_q + i2_q + i3_q + i4_q)
        dx4[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v4_d*i4_d + v4_q*i4_q)) - (self.Dp*self.J)*(omega4 - self.omegag)
        dx4[7] = omega4 - self.omegag
        dx4[8] = (1/self.K)*(self.Qref - (3/2)*(v4_q*i4_d - v4_d*i4_q) - self.Dq*(np.sqrt((v4_d)**2 + (v4_q)**2) - self.Uref))
        dx4[9] = e4*np.cos(phi4) - self.rv*i4_d + omega4*self.L*i4_q - v4_d
        dx4[10] = e4*np.sin(phi4) - self.rv*i4_q - omega4*self.L*i4_d - v4_q
        dx4[11] = self.Kpv*(e4*np.cos(phi4) - self.rv*i4_d + omega4*self.L*i4_q - v4_d) + self.Kiv*eps4_d - omega4*self.C*v4_q - i4_0d
        dx4[12] = self.Kpv*(e4*np.sin(phi4) - self.rv*i4_q - omega4*self.L*i4_d - v4_q) + self.Kiv*eps4_q + omega4*self.C*v4_d - i4_0q

        # Now repeat for node 5
        Rc5 = (self.Rload*Rdl5)/(self.Rload + Rdl5)
        dx5 = np.zeros(len(x5))
        dx5[0] = (1.0/self.L)*(v5_0d - v5_d) + self.omegag*i5_0q
        dx5[1] = (1.0/self.L)*(v5_0q - v5_q) - self.omegag*i5_0d
        dx5[2] = (1.0/self.C)*(i5_0d - i5_d) + self.omegag*v5_q
        dx5[3] = (1.0/self.C)*(i5_0q - i5_q) - self.omegag*v5_d
        dx5[4] = (1.0/self.Lt)*(v5_d - i5_d*self.Rt) + self.omegag*i5_q - (Rc5/self.Lt)*(i1_d + i2_d + i3_d + i4_d)
        dx5[5] = (1.0/self.Lt)*(v5_q - i5_q*self.Rt) - self.omegag*i5_d - (Rc5/self.Lt)*(i1_q + i2_q + i3_q + i4_q)
        dx5[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v5_d*i5_d + v5_q*i5_q)) - (self.Dp*self.J)*(omega5 - self.omegag)
        dx5[7] = omega5 - self.omegag
        dx5[8] = (1/self.K)*(self.Qref - (3/2)*(v5_q*i5_d - v5_d*i5_q) - self.Dq*(np.sqrt((v5_d)**2 + (v5_q)**2) - self.Uref))
        dx5[9] = e5*np.cos(phi5) - self.rv*i5_d + omega5*self.L*i5_q - v5_d
        dx5[10] = e5*np.sin(phi5) - self.rv*i5_q - omega5*self.L*i5_d - v5_q
        dx5[11] = self.Kpv*(e5*np.cos(phi5) - self.rv*i5_d + omega5*self.L*i5_q - v5_d) + self.Kiv*eps5_d - omega5*self.C*v5_q - i5_0d
        dx5[12] = self.Kpv*(e5*np.sin(phi5) - self.rv*i5_q - omega5*self.L*i5_d - v5_q) + self.Kiv*eps5_q + omega5*self.C*v5_d - i5_0q

        # Now repeat for node 5
        Rc6 = (self.Rload*Rdl6)/(self.Rload + Rdl6)
        dx6 = np.zeros(len(x6))
        dx6[0] = (1.0/self.L)*(v6_0d - v6_d) + self.omegag*i6_0q
        dx6[1] = (1.0/self.L)*(v6_0q - v6_q) - self.omegag*i6_0d
        dx6[2] = (1.0/self.C)*(i6_0d - i6_d) + self.omegag*v6_q
        dx6[3] = (1.0/self.C)*(i6_0q - i6_q) - self.omegag*v6_d
        dx6[4] = (1.0/self.Lt)*(v6_d - i6_d*self.Rt) + self.omegag*i6_q - (Rc6/self.Lt)*(i1_d + i2_d + i3_d + i4_d)
        dx6[5] = (1.0/self.Lt)*(v6_q - i6_q*self.Rt) - self.omegag*i6_d - (Rc6/self.Lt)*(i1_q + i2_q + i3_q + i4_q)
        dx6[6] = (1.0/(self.J*self.omegag))*(self.Pref - (3/2)*(v6_d*i6_d + v6_q*i6_q)) - (self.Dp*self.J)*(omega6 - self.omegag)
        dx6[7] = omega6 - self.omegag
        dx6[8] = (1/self.K)*(self.Qref - (3/2)*(v6_q*i6_d - v6_d*i6_q) - self.Dq*(np.sqrt((v6_d)**2 + (v6_q)**2) - self.Uref))
        dx6[9] = e6*np.cos(phi6) - self.rv*i6_d + omega6*self.L*i6_q - v6_d
        dx6[10] = e6*np.sin(phi6) - self.rv*i6_q - omega6*self.L*i6_d - v6_q
        dx6[11] = self.Kpv*(e6*np.cos(phi6) - self.rv*i6_d + omega6*self.L*i6_q - v6_d) + self.Kiv*eps6_d - omega6*self.C*v6_q - i6_0d
        dx6[12] = self.Kpv*(e6*np.sin(phi6) - self.rv*i6_q - omega6*self.L*i6_d - v6_q) + self.Kiv*eps6_q + omega6*self.C*v6_d - i6_0q

        return  np.concatenate([dx1, dx2, dx3, dx4, dx5, dx6])
