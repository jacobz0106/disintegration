import numpy as np
import pandas as pd

class SIR_model(object):
	def __init__(self,initial_conditions = [0.999, 0.001, 0],T = 30,T0 = 10,inc = 0.01, method = 'Heun'):
		"""
		t in [T_0, T_1]
		Synthetic: [0.15,0.45]x [0.05, 0.3], T = 30 # Beta ∼Beta(12, 30), Gamma ∼ Beta(6, 30)
		# real data: surge 1: lambada, beta[0.05, 0.32]X[0.067, 0.25], T = 64 // surge 2:  [0.18, 0.5] x  [.071, .33] , T_0 = 1 
		"""
		self.initial_conditions = initial_conditions
		self.T = T
		self.T0 = T0
		self.method = method
		self.inc = inc

	def function_SIR(self,beta, gamma):
		ls = np.zeros(( ( int((self.T + self.T0)/self.inc) + 1),3))
		ls[0] = self.initial_conditions
		inc = self.inc
		if self.method == 'fw_Euler':
			for i in range(1,len(ls)-1):
				S,I,R = ls[i -1]  
				ls[i][0] = -beta*I*S*inc + S
				ls[i][1] = (beta*I*S - gamma*I)*inc + I
				ls[i][2] = gamma*I*inc + R
		else:
			for i in range(1,len(ls)):
				S,I,R = ls[i -1]  
				intermediate = np.zeros(3)
				intermediate[0] = -beta*I*S*inc + S
				intermediate[1] = (beta*I*S - gamma*I)*inc + I
				intermediate[2] = gamma*I*inc + R

				ls[i][0] = (-beta*I*S - beta*intermediate[1]*intermediate[0])*inc/2 + S
				ls[i][1] = (beta*I*S - gamma*I + beta*intermediate[1]*intermediate[0] - gamma*intermediate[1])*inc/2 + I
				ls[i][2] = (gamma*I + gamma*intermediate[1])*inc/2 + R
		self.df = ls
		return ls

	def quantity_interest(self,para):
		beta, gamma = para
		sol = self.function_SIR(beta, gamma)
		return (sol[-1][1] - sol[ int(self.T0/self.inc) ][1])/self.T

	def gradients(self,para):
		beta, gamma = para
		ls_gamma = np.zeros((self.n, 3))
		ls_gamma[0] = (0, 0, 0)
		ls_beta = np.zeros((self.n, 3))
		ls_beta[0] = (0, 0, 0)
		self.function_SIR(beta, gamma)
		F = self.df
		inc = self.T/self.n
		for i in range(1,self.n):
			S, I, R = F[i-1]

			g1, g2, g3 = ls_gamma[i-1]
			ls_gamma[i][0] = inc*(-beta*I*g1 - beta*S*g2) + ls_gamma[i-1][0]
			ls_gamma[i][1] = inc*(beta*I*g1 + (beta*S - gamma)*g2 - I) + ls_gamma[i-1][1]
			ls_gamma[i][2] = inc*(gamma*g2 + I ) + ls_gamma[i-1][2]

			d1, d2, d3 = ls_beta[i-1]
			ls_beta[i][0] = inc*(-beta*I*d1 - beta*S*d2 - I*S) + ls_beta[i-1][0]
			ls_beta[i][1] = inc*(beta*I*d1 + (beta*S - gamma)*d2 + I*S) + ls_beta[i-1][1]
			ls_beta[i][2] = inc*(gamma*d2) + ls_beta[i-1][2]


		# dS/dbeta, dS/dgamma
		return([ (ls_beta[self.T0 + self.T ][1] - ls_beta[self.T0 ][1])/self.T , (ls_gamma[self.T0 + self.T ][1] - ls_gamma[self.T0 ][1])/self.T ])







