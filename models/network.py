"""
File containing master stability style formulations of various 
generator types. These are implemented as classes: 

- VSM controlled generator with ohmic network load (VSM)
- Droop controlled generator with ohmic load
- VSM controlled generator with complex load
"""

import numpy as np 
from scipy.linalg import block_diag


class VSM(object):
    """
    Class for modelling a network of VSMs, represented
    using a matrix-vector formulation.

    Class methods include functions to compute the 
    variational problem for the MSF.
    """

    def __init__(self):
        """
        Initialisation function.
        """
        super(VSM, self).__init__()
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
        self.Kpc = 10.0 
        self.Kiv = 40 
        self.Kic = 1500 

    def construct_constmatrix(self):
        """
        Method to contruct the matrix of linear 
        terms in the ODEs for each node.
        """
        self.Con = np.zeros((self.m,self.m)) 
        self.Con[0,0] = -self.Kpc/self.L 
        self.Con[0,1] = self.omegag 
        self.Con[0,2] = -(1+self.Kpc*self.Kpv)/self.L  
        self.Con[0,4] = -self.Kpc*self.Kpv*self.rv/self.L 
        self.Con[0,9] = self.Kpc*self.Kiv/self.L 
        self.Con[0,11] = self.Kic/self.L
        self.Con[1,0] = -self.omegag  
        self.Con[1,1] = -self.Kpc/self.L 
        self.Con[1,3] = -(1+self.Kpc*self.Kpv)/self.L  
        self.Con[1,5] = -self.Kpc*self.Kpv*self.rv/self.L 
        self.Con[1,10] = self.Kpc*self.Kiv/self.L 
        self.Con[1,12] = self.Kic/self.L 
        self.Con[2,0] = 1/self.C          
        self.Con[2,3] = self.omegag   
        self.Con[2,4] = -1/self.C 
        self.Con[3,1] = 1/self.C 
        self.Con[3,2] = -self.omegag  
        self.Con[3,5] = -1/self.C  
        self.Con[4,2] = 1/self.Lt 
        self.Con[4,4] = -(self.Rt + self.n*self.Rload)/self.Lt 
        self.Con[4,5] = self.omegag  
        self.Con[5,3] = 1/self.Lt 
        self.Con[5,4] = -self.omegag  
        self.Con[5,5] = -(self.Rt + self.n*self.Rload)/self.Lt  
        self.Con[6,6] = -self.Dp/self.J  
        self.Con[7,6] = 1  
        self.Con[9,2] = -1 
        self.Con[9,4] = -self.rv  
        self.Con[10,3] = -1 
        self.Con[10,5] = -self.rv 
        self.Con[11,0] = -1     
        self.Con[11,2] = -self.Kpv 
        self.Con[11,4] = -self.Kpv*self.rv 
        self.Con[11,9] = self.Kiv  
        self.Con[12,1] = -1 
        self.Con[12,3] = -self.Kpv 
        self.Con[12,5] = -self.Kpv*self.rv  
        self.Con[12,10] = self.Kiv 

    def construct_H(self):
        """
        Method to compute the nodal coupling 
        matrix H.
        """
        self.H = np.zeros((self.m,self.m))
        self.H[4,4] = 1.0 
        self.H[5,5] = 1.0

    def construct_A(self): 
        """
        Method to construct the network's adjacency matrix.
        """
        self.A = np.ones((self.n,self.n)) 
        for i in range(self.n):
            self.A[i,i] = 0.0 

    def construct_D(self): 
        """
        Method to construct the network's degree matrix.
        """
        self.D = np.zeros((self.n,self.n))
        for i in range(self.n):
            self.D[i,i] = self.n - 1.0 

    def compute_z(self): 
        """
        Method to compute the demoninator z in the phasor steady state solution
        of the model, for use in the power calculations.
        """
        Reff = self.n*self.Rload + self.Rt
        self.re_z = 1 - self.L*self.C*self.omega_ref**2 + (self.L*self.Lt*self.omega_ref**2)/(Reff**2 + self.omega_ref**2*self.Lt**2)
        self.im_z = (self.omega_ref*self.L*Reff)/(Reff**2 + self.omega_ref**2*self.Lt**2) 

    def set_pref(self):
        """
        Method to compute the node's reference real power P.
        """
        Reff = self.n*self.Rload + self.Rt
        self.Pref = 3*(self.Uref**2*Reff)/(2*(Reff**2+self.omega_ref**2*self.Lt**2))*(1/(self.re_z**2+self.im_z**2)) 

    def set_qref(self):
        """
        Method to compute the node's reactive real power Q.
        """
        Reff = self.n*self.Rload + self.Rt
        self.Qref = 3*(self.Uref**2*self.omega_ref*self.Lt)/(2*(Reff**2+self.omega_ref**2*self.Lt**2))*(1/(self.re_z**2+self.im_z**2)) 

    def initialise_system(self, **kwargs):
        """
        High-level initialisation method that computes all 
        quantities needed for the ODEs: i.e., the reference
        powers and the coupling and linear constant matrices.
        """
        self.__dict__.update(kwargs)
        self.construct_constmatrix() 
        self.construct_H() 
        self.construct_A() 
        self.construct_D() 
        self.construct_laplacian() 
        self.compute_z() 
        self.set_pref() 
        self.set_qref() 

    def compute_B(self, x): 
        """
        Method to compute the nonlineat terms in the ODEs.
        """
        i0D, i0Q, vD, vQ, iD, iQ, omega, phi, E, _, _, _, _ = x
        B = np.zeros(self.m)
        B[0] = (self.Kpc/self.L)*(self.Kpv*(E*np.cos(phi) + omega*self.L*iQ) - omega*self.C*vQ) - omega*i0Q
        B[1] = (self.Kpc/self.L)*(self.Kpv*(E*np.sin(phi) - omega*self.L*iD) + omega*self.C*vD) + omega*i0D   
        P = (3/2)*(vD*iD + vQ*iQ) 
        Q = (3/2)*(vQ*iD - vD*iQ) 
        Vamp = np.sqrt((vD**2)+(vQ**2)) 
        B[6] = (1/(self.J*self.omegag))*(self.Pref - P) + ((self.Dp*self.omegag)/self.J) 
        B[7] = -self.omegag 
        B[8] = (1/self.K)*(self.Qref - Q - self.Dq*(Vamp - self.Uref)) 
        epsx = E*np.cos(phi) + omega*self.L*iQ 
        epsy = E*np.sin(phi) - omega*self.L*iD  
        B[9] = epsx
        B[10] = epsy  
        B[11] = self.Kpv*epsx - omega*self.C*vQ 
        B[12] = self.Kpv*epsy + omega*self.C*vD 
        return B

    def construct_laplacian(self):
        """
        Method to compute the network's laplacian metrix.
        """
        self.Laplacian = self.D - self.A 

    def T_inv(self, t, omega):
        """
        Method to compute the inverse dq-tranform matrix.
        """
        theta = omega*t
        shift = 2.0*np.pi/3.0
        T = np.zeros((3,3))
        T[0,0] = np.cos(theta)
        T[0,1] = -np.sin(theta)
        T[0,2] = 1
        T[1,0] = np.cos(theta - shift)
        T[1,1] = -np.sin(theta - shift)
        T[1,2] = 1
        T[2,0] = np.cos(theta + shift)
        T[2,1] = -np.sin(theta - shift)
        T[2,2] = 1 
        return T 

    def record_abc(self, t, x): 
        """
        Callback function that can be used to record timeseries
        information of the sinusoidal values of v and i during timestepping.
        """
        self.record_count += 1 
        if (self.record_count % 20) == 0: 
            # print("Time = ", t) 
            self.time.append(t) 

            #Do inverse dq transform 
            signal = np.array((x[2],x[3], 0.0)) 
            abc_signal = np.dot(self.T_inv(t, x[6]), signal) 
            self.va.append(abc_signal[0])
            self.vb.append(abc_signal[1])
            self.vc.append(abc_signal[2])

            #Record power of first node 
            vD, vQ, iD, iQ = x[2], x[3], x[4], x[5]
            self.P.append((3/2)*(vD*iD + vQ*iQ))
            self.Q.append((3/2)*(vQ*iD - vD*iQ))

            #Make combined current and voltage
            icD, icQ = 0.0, 0.0 
            for i in range(0, self.n): 
                vD, vQ, iD, iQ = x[self.m*i:self.m*(i+1)][2], x[self.m*i:self.m*(i+1)][3], x[self.m*i:self.m*(i+1)][4], x[self.m*i:self.m*(i+1)][5]
                icD += iD 
                icQ += iQ  
            self.combinedVD.append(icD*self.Rload)
            self.combinedVQ.append(icQ*self.Rload)
            self.combinedP.append(3*(icD**2 + icQ**2)*self.Rload/2) 


    def F(self, t, x):
        """
        Main ODE function. Creates the RHS of the dynamical equation
        for the network.
        """
        dx = np.zeros(self.n*self.m) 
        for k in range(0,self.n):
            summation = 0 
            for j in range(0, self.n): 
                summation += self.Laplacian[k,j]*np.dot(self.H, x[self.m*j:self.m*(j+1)])
            dx[self.m*k:self.m*(k+1)] = np.dot(self.Con, x[self.m*k:self.m*(k+1)]) +  self.compute_B(x[self.m*k:self.m*(k+1)]) + (self.Rload/self.Lt)*summation
        return dx 

    def Fsteady(self, t, x, **kwargs):
        """
        Returns the decoupled node dynamics ODE.
        """
        self.__dict__.update(kwargs)
        return np.dot(self.Con, x) + self.compute_B(x) 

    def compute_DB(self, x): 
        """
        Function to compute the jacobian matrix of the nonlinear 
        components of the ODE system.
        """
        i0D, i0Q, vD, vQ, iD, iQ, omega, phi, E, _, _, _, _ = x
        DB = np.zeros((self.m,self.m)) 
        DB[0,1] = -omega 
        DB[0,3] = -omega*self.C*self.Kpc/self.L 
        DB[0,5] = self.Kpc*self.Kpv*omega 
        DB[0,6] = self.Kpc*self.Kpv*iQ - (self.Kpc*self.C/self.L)*vQ - i0Q 
        DB[0,7] = -self.Kpc*self.Kpv*E*np.sin(phi)/self.L 
        DB[0,8] = self.Kpc*self.Kpv*np.cos(phi)/self.L 
        DB[1,0] = omega 
        DB[1,2] = self.Kpc*self.C*omega/self.L 
        DB[1,4] = -self.Kpc*self.Kpv*omega 
        DB[1,6] = -self.Kpc*self.Kpv*iD + (self.Kpc*self.C/self.L)*vD + i0D 
        DB[1,7] = self.Kpc*self.Kpv*E*np.cos(phi)/self.L 
        DB[1,8] = self.Kpc*self.Kpv*np.sin(phi)/self.L 
        DB[6,2] = -3*iD/(2*self.J*self.omegag) 
        DB[6,3] = -3*iQ/(2*self.J*self.omegag)
        DB[6,4] = -3*vD/(2*self.J*self.omegag)
        DB[6,5] = -3*vQ/(2*self.J*self.omegag) 
        temp = self.Dq/(self.K*np.sqrt((vD**2) + (vQ**2))) 
        DB[8,2] = 3*iQ/(2*self.K) - temp*vD 
        DB[8,3] = -3*iD/(2*self.K) - temp*vQ 
        DB[8,4] = -3*vQ/(2*self.K) 
        DB[8,5] = 3*vD/(2*self.K) 
        DB[9,5] = omega*self.L 
        DB[9,6] = self.L*iQ 
        DB[9,7] = -E*np.sin(phi) 
        DB[9,8] = np.cos(phi) 
        DB[10,4] = -omega*self.L 
        DB[10,6] = -self.L*iD 
        DB[10,7] = E*np.cos(phi) 
        DB[10,8] = np.sin(phi) 
        DB[11,3] = -omega*self.C 
        DB[11,5] = self.Kpv*omega*self.L 
        DB[11,6] = self.Kpv*self.L*iQ - self.C*vQ 
        DB[11,7] = -self.Kpv*E*np.sin(phi) 
        DB[11,8] = self.Kpv*np.cos(phi) 
        DB[12,2] = omega*self.C
        DB[12,4] = -self.Kpv*omega*self.L 
        DB[12,6] = -self.Kpv*self.L*iD + self.C*vD 
        DB[12,7] = self.Kpv*E*np.cos(phi) 
        DB[12,8] = self.Kpv*np.sin(phi) 
        return DB
        
    def variational(self, _, x, **kwargs):
        """
        Function that returns the variational formulation of the 
        network's dynamics.
        """
        self.__dict__.update(kwargs) 
        return self.Con + self.compute_DB(x) + (self.n*self.Rload/self.Lt)*self.H 

    def jacobian(self, _, x, **kwargs):
        """
        Function to evaluate the Jacobian matrix of the
        variational equation. This is useful for speeding 
        up timestepping.
        """
        self.__dict__.update(kwargs)
        Jacsteady = self.Con + self.compute_DB(x[:self.m]) 
        if len(x) == self.m:
            return Jacsteady
        J = Jacsteady + (self.n*self.Rload/self.Lt)*self.H 
        return block_diag(Jacsteady, J, J, J, J, J, J, J, J, J, J, J, J, J)
    

class DroopDroop(object):
    """
    Class for modelling a network of DGs, represented
    using a matrix-vector formulation, controlled
    by droop controllers.

    Class methods include functions to compute the 
    variational problem for the MSF.
    """
    def __init__(self):
        """
        Initialisation routine.
        """
        super(DroopDroop, self).__init__()
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
        self.m = 12 
        self.Dp = 50
        self.J = 1
        self.kp = 10
        self.K = 0.05
        self.Dq = 100
        self.rv = 0.1
        self.Kpv = 0.5
        self.Kpc = 1
        self.Kiv = 1
        self.Kic = 1500

    def construct_constmatrix(self):
        """
        Method to contruct the matrix of linear 
        terms in the ODEs for each node.
        """
        self.Con = np.zeros((self.m,self.m)) 
        self.Con[0,0] = -self.Kpc/self.L 
        self.Con[0,1] = self.omegag 
        self.Con[0,2] = -(1+self.Kpc*self.Kpv)/self.L  
        self.Con[0,4] = -self.Kpc*self.Kpv*self.rv/self.L 
        self.Con[0,8] = self.Kpc*self.Kiv/self.L 
        self.Con[0,10] = self.Kic/self.L
        self.Con[1,0] = -self.omegag  
        self.Con[1,1] = -self.Kpc/self.L 
        self.Con[1,3] = -(1+self.Kpc*self.Kpv)/self.L  
        self.Con[1,5] = -self.Kpc*self.Kpv*self.rv/self.L 
        self.Con[1,9] = self.Kpc*self.Kiv/self.L 
        self.Con[1,11] = self.Kic/self.L 
        self.Con[2,0] = 1/self.C          
        self.Con[2,3] = self.omegag   
        self.Con[2,4] = -1/self.C 
        self.Con[3,1] = 1/self.C 
        self.Con[3,2] = -self.omegag  
        self.Con[3,5] = -1/self.C  
        self.Con[4,2] = 1/self.Lt 
        self.Con[4,4] = -(self.Rt + self.n*self.Rload)/self.Lt 
        self.Con[4,5] = self.omegag  
        self.Con[5,3] = 1/self.Lt 
        self.Con[5,4] = -self.omegag  
        self.Con[5,5] = -(self.Rt + self.n*self.Rload)/self.Lt  
        self.Con[8,2] = -1 
        self.Con[8,4] = -self.rv  
        self.Con[9,3] = -1 
        self.Con[9,5] = -self.rv 
        self.Con[10,0] = -1     
        self.Con[10,2] = -self.Kpv 
        self.Con[10,4] = -self.Kpv*self.rv 
        self.Con[10,8] = self.Kiv  
        self.Con[11,1] = -1 
        self.Con[11,3] = -self.Kpv 
        self.Con[11,5] = -self.Kpv*self.rv  
        self.Con[11,9] = self.Kiv 

    def construct_H(self):
        """
        Method to compute the nodal coupling 
        matrix H.
        """
        self.H = np.zeros((self.m,self.m))
        self.H[4,4] = 1.0 
        self.H[5,5] = 1.0

    def construct_A(self): 
        """
        Method to construct the network's adjacency matrix.
        """
        self.A = np.ones((self.n,self.n)) 
        for i in range(self.n):
            self.A[i,i] = 0.0 

    def construct_D(self): 
        """
        Method to construct the network's degree matrix.
        """
        self.D = np.zeros((self.n,self.n))
        for i in range(self.n):
            self.D[i,i] = self.n - 1.0 

    def compute_z(self): 
        """
        Method to compute the demoninator z in the phasor steady state solution
        of the model, for use in the power calculations.
        """
        Reff = self.n*self.Rload + self.Rt
        self.re_z = 1 - self.L*self.C*self.omega_ref**2 + (self.L*self.Lt*self.omega_ref**2)/(Reff**2 + self.omega_ref**2*self.Lt**2)
        self.im_z = (self.omega_ref*self.L*Reff)/(Reff**2 + self.omega_ref**2*self.Lt**2) 

    def set_pref(self):
        """
        Method to compute the node's reference real power P.
        """
        Reff = self.n*self.Rload + self.Rt
        self.Pref = 3*(self.Uref**2*Reff)/(2*(Reff**2+self.omega_ref**2*self.Lt**2))*(1/(self.re_z**2+self.im_z**2)) 

    def set_qref(self):
        """
        Method to compute the node's reactive real power Q.
        """
        Reff = self.n*self.Rload + self.Rt
        self.Qref = 3*(self.Uref**2*self.omega_ref*self.Lt)/(2*(Reff**2+self.omega_ref**2*self.Lt**2))*(1/(self.re_z**2+self.im_z**2)) 

    def initialise_system(self, **kwargs):
        """
        High-level initialisation method that computes all 
        quantities needed for the ODEs: i.e., the reference
        powers
        """
        self.__dict__.update(kwargs)
        self.construct_constmatrix() 
        self.construct_H() 
        self.construct_A() 
        self.construct_D() 
        self.construct_laplacian() 
        self.compute_z() 
        self.set_pref() 
        self.set_qref() 

    def compute_B(self, x): 
        """
        Method to compute the nonlineat terms in the ODEs.
        """
        i0D, i0Q, vD, vQ, iD, iQ, phi, E, _, _, _, _ = x
        P = (3/2)*(vD*iD + vQ*iQ) 
        Q = (3/2)*(vQ*iD - vD*iQ) 
        omega = self.omegag - self.kp*(P - self.Pref) 
        B = np.zeros(self.m)
        B[0] = (self.Kpc/self.L)*(self.Kpv*(E*np.cos(phi) + omega*self.L*iQ) - omega*self.C*vQ) - omega*i0Q
        B[1] = (self.Kpc/self.L)*(self.Kpv*(E*np.sin(phi) - omega*self.L*iD) + omega*self.C*vD) + omega*i0D  
        B[6] = omega
        Vamp = np.sqrt((vD**2)+(vQ**2)) 
        B[7] = (1/self.K)*(self.Qref - Q - self.Dq*(Vamp - self.Uref)) 
        epsx = E*np.cos(phi) + omega*self.L*iQ 
        epsy = E*np.sin(phi) - omega*self.L*iD  
        B[8] = epsx
        B[9] = epsy  
        B[10] = self.Kpv*epsx - omega*self.C*vQ 
        B[11] = self.Kpv*epsy + omega*self.C*vD 
        return B 

    def construct_laplacian(self):
        """
        Method to compute the network's laplacian metrix.
        """
        self.Laplacian = self.D - self.A 

    def T_inv(self, theta):
        """
        Method to compute the inverse dq-tranform matrix.
        """
        shift = 2.0*np.pi/3.0
        T = np.zeros((3,3))
        T[0,0] = np.cos(theta)
        T[0,1] = -np.sin(theta)
        T[0,2] = 1
        T[1,0] = np.cos(theta - shift)
        T[1,1] = -np.sin(theta - shift)
        T[1,2] = 1
        T[2,0] = np.cos(theta + shift)
        T[2,1] = -np.sin(theta - shift)
        T[2,2] = 1 
        return T 

    def record_abc(self, t, x): 
        """
        Callback function that can be used to record timeseries
        information of the sinusoidal values of v and i during timestepping.
        """
        self.record_count += 1 
        if (self.record_count % 50) == 0: 
            print("Time = ", t) 
            self.time.append(t) 

            #Do inverse dq transform 
            signal = np.array((x[2],x[3], 0.0)) 
            abc_signal = np.dot(self.T_inv(x[6]), signal) 
            self.va.append(abc_signal[0])
            self.vb.append(abc_signal[1])
            self.vc.append(abc_signal[2])

            #Record power of first node 
            vD, vQ, iD, iQ = x[2], x[3], x[4], x[5]
            self.P.append((3/2)*(vD*iD + vQ*iQ))
            self.Q.append((3/2)*(vQ*iD - vD*iQ))

            #Make combined current and voltage
            icD, icQ = 0.0, 0.0 
            for i in range(0, self.n): 
                vD, vQ, iD, iQ = x[self.m*i:self.m*(i+1)][2], x[self.m*i:self.m*(i+1)][3], x[self.m*i:self.m*(i+1)][4], x[self.m*i:self.m*(i+1)][5]
                icD += iD 
                icQ += iQ  
            self.combinedVD.append(icD*self.Rload)
            self.combinedVQ.append(icQ*self.Rload)
            self.combinedP.append(3*(icD**2 + icQ**2)*self.Rload/2) 


    def F(self, t, x):
        """
        Main ODE function. Creates the RHS of the dynamical equation
        for the network.
        """
        # self.record_abc(t, x) 
        dx = np.zeros(self.n*self.m) 
        for k in range(0,self.n):
            summation = 0 
            for j in range(0, self.n): 
                summation += self.Laplacian[k,j]*np.dot(self.H, x[self.m*j:self.m*(j+1)])
            dx[self.m*k:self.m*(k+1)] = np.dot(self.Con, x[self.m*k:self.m*(k+1)]) +  self.compute_B(x[self.m*k:self.m*(k+1)]) + (self.Rload/self.Lt)*summation
        return dx 

    def Fsteady(self, t, x, **kwargs):
        """
        Returns the decoupled node dynamics ODE.
        """
        self.__dict__.update(kwargs)
        return np.dot(self.Con, x) + self.compute_B(x) 

    def compute_DB(self, x): 
        """
        Function to compute the jacobian matrix of the nonlinear 
        components of the ODE system.
        """
        i0D, i0Q, vD, vQ, iD, iQ, phi, E, _, _, _, _ = x
        P = (3/2)*(vD*iD + vQ*iQ) 
        Q = (3/2)*(vQ*iD - vD*iQ) 
        omega = self.omegag + self.kp*(P - self.Pref)
        DB = np.zeros((self.m,self.m)) 
        DB[0,1] = -omega 
        DB[0,2] = (3/2)*self.kp*iD*(self.Kpc*self.Kpv*iQ - (self.Kpc/self.L)*self.C*vQ - i0Q)
        DB[0,3] = -(self.Kpc*omega*self.C/self.L) + (3/2)*self.kp*iQ*(self.Kpc*self.Kpv*iQ - (self.Kpc/self.L)*self.C*vQ - i0Q)
        DB[0,4] = (3/2)*self.kp*vD*(self.Kpc*self.Kpv*iQ - (self.Kpc/self.L)*self.C*vQ - i0Q)
        DB[0,5] = self.Kpc*self.Kpv*omega + (3/2)*self.kp*vQ*(self.Kpc*self.Kpv*iQ - (self.Kpc/self.L)*self.C*vQ - i0Q)
        DB[0,6] = -self.Kpc*self.Kpv*E*np.sin(phi)/self.L
        DB[0,7] = self.Kpc*self.Kpv*np.cos(phi)/self.L 
        DB[1,0] = omega 
        DB[1,2] = (self.Kpc*self.C*omega/self.L) + (3/2)*self.kp*iD*(-self.Kpc*self.Kpv*iD + (self.Kpc/self.L)*self.C*vD + i0D)
        DB[1,3] = (3/2)*self.kp*iQ*(-self.Kpc*self.Kpv*iD + (self.Kpc/self.L)*self.C*vD + i0D)
        DB[1,4] = -self.Kpc*self.Kpv*omega + (3/2)*self.kp*vD*(-self.Kpc*self.Kpv*iD + (self.Kpc/self.L)*self.C*vD + i0D)
        DB[1,5] = (3/2)*self.kp*vQ*(-self.Kpc*self.Kpv*iD + (self.Kpc/self.L)*self.C*vD + i0D)
        DB[1,6] = self.Kpc*self.Kpv*E*np.cos(phi)/self.L 
        DB[1,7] = self.Kpc*self.Kpv*np.sin(phi)/self.L 
        DB[6,2] = (3/2)*self.kp*iD
        DB[6,3] = (3/2)*self.kp*iQ
        DB[6,4] = (3/2)*self.kp*vD
        DB[6,5] = (3/2)*self.kp*vQ
        temp = self.Dq/(self.K*np.sqrt((vD**2) + (vQ**2))) 
        DB[7,2] = 3*iQ/(2*self.K) - temp*vD 
        DB[7,3] = -3*iD/(2*self.K) - temp*vQ 
        DB[7,4] = -3*vQ/(2*self.K) 
        DB[7,5] = 3*vD/(2*self.K) 
        DB[8,2] = (3/2)*self.kp*iD*self.L*iQ
        DB[8,3] = (3/2)*self.kp*iQ*self.L*iQ
        DB[8,4] = (3/2)*self.kp*vD*self.L*iQ 
        DB[8,5] = (3/2)*self.kp*vQ*self.L*iQ + omega*self.L
        DB[8,6] = -E*np.sin(phi)
        DB[8,7] = np.cos(phi)
        DB[9,2] = -(3/2)*self.kp*iD*self.L*iD
        DB[9,3] = -(3/2)*self.kp*iQ*self.L*iD
        DB[9,4] = -(3/2)*self.kp*vD*self.L*iD - omega*self.L
        DB[9,5] = -(3/2)*self.kp*vQ*self.L*iD
        DB[9,6] = E*np.cos(phi)
        DB[9,7] = np.sin(phi)
        DB[10,2] = (3/2)*self.kp*iD*(self.Kpv*self.L*iQ - self.C*vQ)
        DB[10,3] = (3/2)*self.kp*iQ*(self.Kpv*self.L*iQ - self.C*vQ) - omega*self.C
        DB[10,4] = (3/2)*self.kp*vD*(self.Kpv*self.L*iQ - self.C*vQ)
        DB[10,5] = (3/2)*self.kp*vQ*(self.Kpv*self.L*iQ - self.C*vQ) + self.Kpv*omega*self.L
        DB[10,6] = -self.Kpv*E*np.sin(phi)
        DB[10,7] = self.Kpv*np.cos(phi)
        DB[11,2] = omega*self.C + (3/2)*self.kp*iD*(self.C*vD - self.Kpv*self.L*iD) 
        DB[11,3] = (3/2)*self.kp*iQ*(self.C*vD - self.Kpv*self.L*iD) 
        DB[11,4] = (3/2)*self.kp*vD*(self.C*vD - self.Kpv*self.L*iD) - self.Kpv*omega*self.L
        DB[11,5] = (3/2)*self.kp*vQ*(self.C*vD - self.Kpv*self.L*iD) 
        DB[11,6] = self.Kpv*E*np.cos(phi)
        DB[11,7] = self.Kpv*np.sin(phi)
        return DB
        
    def variational(self, _, x, **kwargs):
        """
        Function that returns the variational formulation of the 
        network's dynamics.
        """
        self.__dict__.update(kwargs) 
        return self.Con + self.compute_DB(x) + (self.n*self.Rload/self.Lt)*self.H 

    def jacobian(self, _, x, **kwargs):
        """
        Function to evaluate the Jacobian matrix of the
        variational equation. This is useful for speeding 
        up timestepping.
        """
        self.__dict__.update(kwargs)
        Jacsteady = self.Con + self.compute_DB(x[:self.m]) 
        if len(x) == self.m:
            return Jacsteady
        J = Jacsteady + (self.n*self.Rload/self.Lt)*self.H 
        return block_diag(Jacsteady, J, J, J, J, J, J, J, J, J, J, J, J)


class VSM_complexload(object):
    """
    Class for modelling a network of VSMs, represented
    using a matrix-vector formulation, with complex loads.

    Class methods include functions to compute the 
    variational problem for the MSF.
    """
    def __init__(self):
        """
        Initialisation routine.
        """
        super(VSM_complexload, self).__init__()
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
        self.Ld = 0.0 
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

    def construct_constmatrix(self):
        """
        Method to contruct the matrix of linear 
        terms in the ODEs for each node.
        """
        self.Con = np.zeros((self.m,self.m)) 
        self.sigma = 1.0/(self.Lt+self.n*self.Ld) 
        self.Con[0,0] = -self.Kpc/self.L 
        self.Con[0,1] = self.omegag 
        self.Con[0,2] = -(1+self.Kpc*self.Kpv)/self.L  
        self.Con[0,4] = -self.Kpc*self.Kpv*self.rv/self.L 
        self.Con[0,9] = self.Kpc*self.Kiv/self.L 
        self.Con[0,11] = self.Kic/self.L
        self.Con[1,0] = -self.omegag  
        self.Con[1,1] = -self.Kpc/self.L 
        self.Con[1,3] = -(1+self.Kpc*self.Kpv)/self.L  
        self.Con[1,5] = -self.Kpc*self.Kpv*self.rv/self.L 
        self.Con[1,10] = self.Kpc*self.Kiv/self.L 
        self.Con[1,12] = self.Kic/self.L 
        self.Con[2,0] = 1/self.C          
        self.Con[2,3] = self.omegag   
        self.Con[2,4] = -1/self.C 
        self.Con[3,1] = 1/self.C 
        self.Con[3,2] = -self.omegag  
        self.Con[3,5] = -1/self.C  
        self.Con[4,2] = self.sigma
        self.Con[4,4] = -(self.Rt + self.n*self.Rload)*self.sigma 
        self.Con[4,5] = self.omegag  
        self.Con[5,3] = self.sigma
        self.Con[5,4] = -self.omegag  
        self.Con[5,5] = -(self.Rt + self.n*self.Rload)*self.sigma  
        self.Con[6,6] = -self.Dp/self.J  
        self.Con[7,6] = 1  
        self.Con[9,2] = -1 
        self.Con[9,4] = -self.rv  
        self.Con[10,3] = -1 
        self.Con[10,5] = -self.rv 
        self.Con[11,0] = -1     
        self.Con[11,2] = -self.Kpv 
        self.Con[11,4] = -self.Kpv*self.rv 
        self.Con[11,9] = self.Kiv  
        self.Con[12,1] = -1 
        self.Con[12,3] = -self.Kpv 
        self.Con[12,5] = -self.Kpv*self.rv  
        self.Con[12,10] = self.Kiv 

    def construct_H(self):
        """
        Method to compute the nodal coupling 
        matrix H.
        """
        self.H = np.zeros((self.m,self.m))
        self.H[4,2] = self.Ld 
        self.H[5,3] = self.Ld 
        self.H[4,4] = self.Lt*self.Rload - self.Rt*self.Ld
        self.H[5,5] = self.Lt*self.Rload - self.Rt*self.Ld

    def construct_A(self): 
        """
        Method to construct the network's adjacency matrix.
        """
        self.A = np.ones((self.n,self.n)) 
        for i in range(self.n):
            self.A[i,i] = 0.0 

    def construct_D(self): 
        """
        Method to construct the network's degree matrix.
        """
        self.D = np.zeros((self.n,self.n))
        for i in range(self.n):
            self.D[i,i] = self.n - 1.0 

    def compute_z(self): 
        """
        Method to compute the demoninator z in the phasor steady state solution
        of the model, for use in the power calculations.
        """
        Reff = self.n*self.Rload + self.Rt
        Leff = self.n*self.Ld + self.Lt 
        self.re_z = 1 - self.L*self.C*self.omega_ref**2 + (self.L*Leff*self.omega_ref**2)/(Reff**2 + self.omega_ref**2*Leff**2)
        self.im_z = (self.omega_ref*self.L*Reff)/(Reff**2 + self.omega_ref**2*Leff**2) 

    def set_pref(self):
        """
        Method to compute the node's reference real power P.
        """
        Reff = self.n*self.Rload + self.Rt
        Leff = self.n*self.Ld + self.Lt 
        self.Pref = 3*(self.Uref**2*Reff)/(2*(Reff**2+self.omega_ref**2*Leff**2))*(1/(self.re_z**2+self.im_z**2)) 

    def set_qref(self):
        """
        Method to compute the node's reactive real power Q.
        """
        Reff = self.n*self.Rload + self.Rt
        Leff = self.n*self.Ld + self.Lt 
        self.Qref = 3*(self.Uref**2*self.omega_ref*Leff)/(2*(Reff**2+self.omega_ref**2*Leff**2))*(1/(self.re_z**2+self.im_z**2)) 

    def initialise_system(self, **kwargs):
        """
        High-level initialisation method that computes all 
        quantities needed for the ODEs: i.e., the reference
        powers 
        """
        self.__dict__.update(kwargs)
        self.construct_constmatrix() 
        self.construct_H() 
        self.construct_A() 
        self.construct_D() 
        self.construct_laplacian() 
        self.compute_z() 
        self.set_pref() 
        self.set_qref() 

    def compute_B(self, x): 
        """
        Method to compute the nonlineat terms in the ODEs.
        """
        i0D, i0Q, vD, vQ, iD, iQ, omega, phi, E, _, _, _, _ = x
        B = np.zeros(self.m)
        B[0] = (self.Kpc/self.L)*(self.Kpv*(E*np.cos(phi) + omega*self.L*iQ) - omega*self.C*vQ) - omega*i0Q
        B[1] = (self.Kpc/self.L)*(self.Kpv*(E*np.sin(phi) - omega*self.L*iD) + omega*self.C*vD) + omega*i0D   
        P = (3/2)*(vD*iD + vQ*iQ) 
        Q = (3/2)*(vQ*iD - vD*iQ) 
        Vamp = np.sqrt((vD**2)+(vQ**2)) 
        B[6] = (1/(self.J*self.omegag))*(self.Pref - P) + ((self.Dp*self.omegag)/self.J) 
        B[7] = -self.omegag 
        B[8] = (1/self.K)*(self.Qref - Q - self.Dq*(Vamp - self.Uref)) 
        epsx = E*np.cos(phi) + omega*self.L*iQ 
        epsy = E*np.sin(phi) - omega*self.L*iD  
        B[9] = epsx
        B[10] = epsy  
        B[11] = self.Kpv*epsx - omega*self.C*vQ 
        B[12] = self.Kpv*epsy + omega*self.C*vD 
        return B 

    def construct_laplacian(self):
        """
        Method to compute the network's laplacian metrix.
        """
        self.Laplacian = self.D - self.A 

    def T_inv(self, t, omega):
        """
        Method to compute the inverse dq-tranform matrix.
        """
        theta = omega*t
        shift = 2.0*np.pi/3.0
        T = np.zeros((3,3))
        T[0,0] = np.cos(theta)
        T[0,1] = -np.sin(theta)
        T[0,2] = 1
        T[1,0] = np.cos(theta - shift)
        T[1,1] = -np.sin(theta - shift)
        T[1,2] = 1
        T[2,0] = np.cos(theta + shift)
        T[2,1] = -np.sin(theta - shift)
        T[2,2] = 1 
        return T 

    def record_abc(self, t, x): 
        """
        Callback function that can be used to record timeseries
        information of the sinusoidal values of v and i during timestepping.
        """
        self.record_count += 1 
        if (self.record_count % 50) == 0: 
            print("Time = ", t) 
            self.time.append(t) 

            #Do inverse dq transform 
            signal = np.array((x[2],x[3], 0.0)) 
            abc_signal = np.dot(self.T_inv(t, x[6]), signal) 
            self.va.append(abc_signal[0])
            self.vb.append(abc_signal[1])
            self.vc.append(abc_signal[2])

            #Record power of first node 
            vD, vQ, iD, iQ = x[2], x[3], x[4], x[5]
            self.P.append((3/2)*(vD*iD + vQ*iQ))
            self.Q.append((3/2)*(vQ*iD - vD*iQ))

            #Make combined current and voltage
            icD, icQ = 0.0, 0.0 
            for i in range(0, self.n): 
                vD, vQ, iD, iQ = x[self.m*i:self.m*(i+1)][2], x[self.m*i:self.m*(i+1)][3], x[self.m*i:self.m*(i+1)][4], x[self.m*i:self.m*(i+1)][5]
                icD += iD 
                icQ += iQ  
            self.combinedVD.append(icD*self.Rload)
            self.combinedVQ.append(icQ*self.Rload)
            self.combinedP.append(3*(icD**2 + icQ**2)*self.Rload/2) 

    def F(self, t, x):
        """
        Main ODE function. Creates the RHS of the dynamical equation
        for the network.
        """
        # print(t)
        # self.record_abc(t, x) 
        self.sigma = 1.0/(self.Lt+self.n*self.Ld) 
        dx = np.zeros(self.n*self.m) 
        # if np.abs(t-0.5) < 1e-3: 
        #     x[6] = 2*np.pi*20
        for k in range(0,self.n):
            summation = 0 
            for j in range(0, self.n): 
                summation += self.Laplacian[k,j]*np.dot(self.H, x[self.m*j:self.m*(j+1)])
            dx[self.m*k:self.m*(k+1)] = np.dot(self.Con, x[self.m*k:self.m*(k+1)]) +  self.compute_B(x[self.m*k:self.m*(k+1)]) + (self.sigma/self.Lt)*summation
        return dx 

    def Fsteady(self, t, x, **kwargs):
        """
        Returns the decoupled node dynamics ODE.
        """
        self.__dict__.update(kwargs)
        # print(t)
        return np.dot(self.Con, x) + self.compute_B(x) 

    def compute_DB(self, x): 
        """
        Function to compute the jacobian matrix of the nonlinear 
        components of the ODE system.
        """
        i0D, i0Q, vD, vQ, iD, iQ, omega, phi, E, _, _, _, _ = x
        DB = np.zeros((self.m,self.m)) 
        DB[0,1] = -omega 
        DB[0,3] = -omega*self.C*self.Kpc/self.L 
        DB[0,5] = self.Kpc*self.Kpv*omega 
        DB[0,6] = self.Kpc*self.Kpv*iQ - (self.Kpc*self.C/self.L)*vQ - i0Q 
        DB[0,7] = -self.Kpc*self.Kpv*E*np.sin(phi)/self.L 
        DB[0,8] = self.Kpc*self.Kpv*np.cos(phi)/self.L 
        DB[1,0] = omega 
        DB[1,2] = self.Kpc*self.C*omega/self.L 
        DB[1,4] = -self.Kpc*self.Kpv*omega 
        DB[1,6] = -self.Kpc*self.Kpv*iD + (self.Kpc*self.C/self.L)*vD + i0D 
        DB[1,7] = self.Kpc*self.Kpv*E*np.cos(phi)/self.L 
        DB[1,8] = self.Kpc*self.Kpv*np.sin(phi)/self.L 
        DB[6,2] = -3*iD/(2*self.J*self.omegag) 
        DB[6,3] = -3*iQ/(2*self.J*self.omegag)
        DB[6,4] = -3*vD/(2*self.J*self.omegag)
        DB[6,5] = -3*vQ/(2*self.J*self.omegag) 
        temp = self.Dq/(self.K*np.sqrt((vD**2) + (vQ**2))) 
        DB[8,2] = 3*iQ/(2*self.K) - temp*vD 
        DB[8,3] = -3*iD/(2*self.K) - temp*vQ 
        DB[8,4] = -3*vQ/(2*self.K) 
        DB[8,5] = 3*vD/(2*self.K) 
        DB[9,5] = omega*self.L 
        DB[9,6] = self.L*iQ 
        DB[9,7] = -E*np.sin(phi) 
        DB[9,8] = np.cos(phi) 
        DB[10,4] = -omega*self.L 
        DB[10,6] = -self.L*iD 
        DB[10,7] = E*np.cos(phi) 
        DB[10,8] = np.sin(phi) 
        DB[11,3] = -omega*self.C 
        DB[11,5] = self.Kpv*omega*self.L 
        DB[11,6] = self.Kpv*self.L*iQ - self.C*vQ 
        DB[11,7] = -self.Kpv*E*np.sin(phi) 
        DB[11,8] = self.Kpv*np.cos(phi) 
        DB[12,2] = omega*self.C
        DB[12,4] = -self.Kpv*omega*self.L 
        DB[12,6] = -self.Kpv*self.L*iD + self.C*vD 
        DB[12,7] = self.Kpv*E*np.cos(phi) 
        DB[12,8] = self.Kpv*np.sin(phi) 
        return DB
        
    def variational(self, _, x, **kwargs):
        """
        Function that returns the variational formulation of the 
        network's dynamics.
        """
        self.__dict__.update(kwargs) 
        self.sigma = 1.0/(self.Lt+self.n*self.Ld) 
        return self.Con + self.compute_DB(x) + (self.n*self.sigma/self.Lt)*self.H 

    def jacobian(self, _, x, **kwargs):
        """
        Function to evaluate the Jacobian matrix of the
        variational equation. This is useful for speeding 
        up timestepping.
        """
        self.__dict__.update(kwargs)
        self.sigma = 1.0/(self.Lt+self.n*self.Ld) 
        Jacsteady = self.Con + self.compute_DB(x[:self.m]) 
        if len(x) == self.m:
            return Jacsteady
        J = Jacsteady + (self.n*self.sigma/self.Lt)*self.H 
        return block_diag(Jacsteady, J, J, J, J, J, J, J, J, J, J, J, J, J) 