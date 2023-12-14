#! usr/bin/env python 

"""
A file containing implementation of the steady state model for an LCLR network. 
Phasor implmentation. 

These steady state models are used to compute power values.
"""

import numpy as np 

class LCLRsinglephase(object):
	"""Steady state phasor model for LCLRsinglephase"""
	def __init__(self):
		super(LCLRsinglephase, self).__init__()
		self.L = 39.e-3 
		self.C = 2.2e-6 
		self.omega = 2*np.pi*50 
		self.R = 60.0 
		self.Ld = 1e-3 
		self.V0 = 400/np.sqrt(3) 

	def compute_z(self): 
		self.re_z = 1 - self.L*self.C*self.omega**2 + (self.L*self.Ld*self.omega**2)/(self.R**2 + self.omega**2*self.Ld**2)
		self.im_z = (self.omega*self.L*self.R)/(self.R**2 + self.omega**2*self.Ld**2) 

	def voltage_phase(self):
		self.phi = -np.arctan(self.im_z/self.re_z)

	def current_phase(self): 
		self.dphi = -np.arctan(self.im_z/self.re_z) - np.arctan(self.omega*self.Ld/self.R) 

	def compute_voltage(self, **kwargs):  
		self.__dict__.update(kwargs) 
		self.compute_z() 
		a = 1/np.sqrt(self.re_z**2 + self.im_z**2) 
		return a*self.V0 

	def compute_current(self, **kwargs):
		self.__dict__.update(kwargs)
		self.compute_z() 
		a = 1/np.sqrt(self.re_z**2 + self.im_z**2) 
		da = a*(1/np.sqrt(self.R**2+(self.omega*self.Ld)**2))
		return da*self.V0 

	def real_power(self, **kwargs):
		self.__dict__.update(kwargs)
		self.compute_z() 
		return (self.V0**2*self.R)/(2*(self.R**2+self.omega**2*self.Ld**2))*(1/(self.re_z**2+self.im_z**2))

	def reactive_power(self, **kwargs):
		self.__dict__.update(kwargs)
		self.compute_z() 
		return (self.V0**2*self.omega*self.Ld)/(2*(self.R**2+self.omega**2*self.Ld**2))*(1/(self.re_z**2+self.im_z**2)) 
	

class LCLRsinglephase_localload(object):
	"""Steady state phasor model for LCLRsinglephase"""
	def __init__(self):
		super(LCLRsinglephase_localload, self).__init__()
		self.L = 39.e-3 
		self.C = 2.2e-6 
		self.omega = 2*np.pi*50 
		self.R = 60.0 
		self.Lt = 1e-3 
		self.V0 = 400/np.sqrt(3)
		self.Rdl = 50.0
		self.Rt = 0.06

	def compute_z(self): 

		self.re_z = (1/((self.Lt*self.Lt*self.omega*self.omega + self.R**2)*self.Rdl))*(self.Lt*self.Lt*self.omega*self.omega*(2*self.Rdl + self.Rt) + self.R*(self.Rdl*self.Rt + self.R*(self.Rdl + self.Rt)) - self.L*self.omega*self.omega*(-self.Lt*self.Rdl + self.C*self.Lt*self.Lt*self.omega*self.omega*(2*self.Rdl + self.Rt) + self.C*self.R*(self.Rdl*self.Rt + self.R*(self.Rdl + self.Rt))))

		self.im_z = (1/((self.Lt*self.Lt*self.omega*self.omega + self.R**2)*self.Rdl))*self.omega*(self.Lt*(self.Lt*self.Lt*self.omega*self.omega + self.R*self.R + self.R*self.Rdl - self.Rdl*self.Rt) + self.L*(self.Lt*self.Lt*self.omega*self.omega - self.C*self.Lt*self.Lt*self.Lt*self.omega*self.omega*self.omega*self.omega + self.R*(self.R + self.Rdl) - self.C*self.Lt*self.omega*self.omega*(self.R*self.R + self.R*self.Rdl - self.Rdl*self.Rt)))

	def real_power(self, **kwargs):
		self.__dict__.update(kwargs)
		self.compute_z() 
		return (self.V0**2*self.R)/(2*(self.R**2+self.omega**2*self.Lt**2))*(1/(self.re_z**2+self.im_z**2))

	def reactive_power(self, **kwargs):
		self.__dict__.update(kwargs)
		self.compute_z() 
		return (self.V0**2*self.omega*self.Ld)/(2*(self.R**2+self.omega**2*self.Lt**2))*(1/(self.re_z**2+self.im_z**2)) 








