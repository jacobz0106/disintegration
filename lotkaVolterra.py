import numpy as np
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import solve
import matplotlib.pyplot as plt


class lotkaVolterra(object):
	def __init__(self,initial_conditions = [10,5,2],self_interacting_terms =[0.5,0.5,0.5],T= 10,n = 1000, method = 'Heun'):
		'''
		r: [.1, 2]
		alpha_ii: 1
		'''
		self.method = method
		self.df =[]
		self.initial_conditions = initial_conditions
		self.self_interacting_terms = self_interacting_terms
		self.n = n
		self.inc = T / n

	def function_z(self, R,interaction_terms):
		""" 
		initial_conditions: tuple/array of 3
		self_interacting_terms: number of line spaces used to evaluate the intergral

		"""
		ls = np.zeros((self.n,3))
		ls[0] = self.initial_conditions
		#alpha[0,0], alpha[1,1], alpha[2,2] = self.self_interacting_terms
		r1, r2,r3 = R
		alpha = interaction_terms
		t = self.inc
		# loop from t to T-t
		if self.method == 'fw_Euler':
			for i in range(1,self.n):        
				ls_intermediate = np.zeros(3)     
				ls_intermediate[0] = t*(r1*ls[i-1][0]*(1 - alpha[0,0]*ls[i-1][0] - alpha[0,1]*ls[i-1][1] - alpha[0,2]*ls[i-1][2])) + ls[i-1][0]
				ls_intermediate[1] = t*(r2*ls[i-1][1]*(1 - alpha[1,0]*ls[i-1][0] - alpha[1,1]*ls[i-1][1] - alpha[1,2]*ls[i-1][2])) + ls[i-1][1] 
				ls_intermediate[2] = t*(r3*ls[i-1][2]*(1 - alpha[2,0]*ls[i-1][0] - alpha[2,1]*ls[i-1][1] - alpha[2,2]*ls[i-1][2])) + ls[i-1][2]
				ls[i] = ls_intermediate
		else:
			for i in range(1,self.n): 
				ls_intermediate = np.zeros(3)  
				ls_intermediate[0] = t*(r1*ls[i-1][0]*(1 - alpha[0,0]*ls[i-1][0] - alpha[0,1]*ls[i-1][1] - alpha[0,2]*ls[i-1][2])) + ls[i-1][0]
				ls_intermediate[1] = t*(r2*ls[i-1][1]*(1 - alpha[1,0]*ls[i-1][0] - alpha[1,1]*ls[i-1][1] - alpha[1,2]*ls[i-1][2])) + ls[i-1][1] 
				ls_intermediate[2] = t*(r3*ls[i-1][2]*(1 - alpha[2,0]*ls[i-1][0] - alpha[2,1]*ls[i-1][1] - alpha[2,2]*ls[i-1][2])) + ls[i-1][2]

				l1 = t/2*(r1*ls[i-1][0]*(1 - alpha[0,0]*ls[i-1][0] - alpha[0,1]*ls[i-1][1] - alpha[0,2]*ls[i-1][2]) + r1*ls_intermediate[0]*(1 - alpha[0,0]*ls_intermediate[0] - alpha[0,1]*ls_intermediate[1] - alpha[0,2]*ls_intermediate[2])) + ls[i-1][0] 
				l2 = t/2*(r2*ls[i-1][1]*(1 - alpha[1,0]*ls[i-1][0] - alpha[1,1]*ls[i-1][1] - alpha[1,2]*ls[i-1][2]) + r2*ls_intermediate[1]*(1 - alpha[1,0]*ls_intermediate[0] - alpha[1,1]*ls_intermediate[1] - alpha[1,2]*ls_intermediate[2])) + ls[i-1][1] 
				l3 = t/2*(r3*ls[i-1][2]*(1 - alpha[2,0]*ls[i-1][0] - alpha[2,1]*ls[i-1][1] - alpha[2,2]*ls[i-1][2]) + r3*ls_intermediate[2]*(1 - alpha[2,0]*ls_intermediate[0] - alpha[2,1]*ls_intermediate[1] - alpha[2,2]*ls_intermediate[2])) + ls[i-1][2] 	
				ls[i] = [l1,l2,l3]
		self.df = ls
		return ls

	def two_step_forward_gradient(self, R, interaction_terms, sensitivity_source):
		"""
		Generalized two-step forward method for gradient computation.
		
		Parameters:
		- R: list of r1, r2, r3
		- interaction_terms: 3x3 matrix
		- sensitivity_source: string like 'r1', 'alpha12', etc.
		"""
		self.function_z(R, interaction_terms)
		ls = np.zeros((self.n, 3))
		ls[0] = (0, 0, 0)
		r1, r2, r3 = R
		alpha = interaction_terms
		F = self.df

		for i in range(1, self.n):
			z1, z2, z3 = F[i-1]
			y1, y2, y3 = ls[i-1]

			# base derivatives
			dz = [z1, z2, z3]
			dy = [y1, y2, y3]
			dr = [r1, r2, r3]

			def rhs(z, y):
				g = np.zeros(3)
				# compute gradient equations
				for idx in range(3):
					zi = z[idx]
					ri = dr[idx]
					terms = 0
					for j in range(3):
						if j == idx:
							terms += 2 * alpha[idx, j] * z[j]
						else:
							terms += alpha[idx, j] * z[j]

					g[idx] = self.inc * (ri * (1 - terms) * y[idx])
					for j in range(3):
						if j != idx:
							g[idx] -= self.inc * alpha[idx, j] * ri * z[idx] * y[j]

				# insert source terms
				if sensitivity_source == 'r1':
					g[0] += self.inc * z[0] * (1 - alpha[0,0]*z[0] - alpha[0,1]*z[1] - alpha[0,2]*z[2])
				elif sensitivity_source == 'r2':
					g[1] += self.inc * z[1] * (1 - alpha[1,0]*z[0] - alpha[1,1]*z[1] - alpha[1,2]*z[2])
				elif sensitivity_source == 'r3':
					g[2] += self.inc * z[2] * (1 - alpha[2,0]*z[0] - alpha[2,1]*z[1] - alpha[2,2]*z[2])
				elif sensitivity_source == 'alpha12':
					g[0] -= self.inc * r1 * z[0] * z[1]
				elif sensitivity_source == 'alpha13':
					g[0] -= self.inc * r1 * z[0] * z[2]
				elif sensitivity_source == 'alpha21':
					g[1] -= self.inc * r2 * z[1] * z[0]
				elif sensitivity_source == 'alpha23':
					g[1] -= self.inc * r2 * z[1] * z[2]
				elif sensitivity_source == 'alpha31':
					g[2] -= self.inc * r3 * z[2] * z[0]
				elif sensitivity_source == 'alpha32':
					g[2] -= self.inc * r3 * z[2] * z[1]
				return g

			# Predictor step
			pred = rhs(dz, dy)
			y_pred = [y + p for y, p in zip(dy, pred)]

			# Corrector step
			z_next = F[i]
			corr = rhs(z_next, y_pred)

			# Final update
			ls[i] = [y + 0.5 * (dp + dc) for y, dp, dc in zip(dy, pred, corr)]

		return ls[-1][-1]

	def gradient_r1(self, R, interaction_terms):
		return self.two_step_forward_gradient(R, interaction_terms, 'r1')

	def gradient_r2(self, R, interaction_terms):
		return self.two_step_forward_gradient(R, interaction_terms, 'r2')

	def gradient_r3(self, R, interaction_terms):
		return self.two_step_forward_gradient(R, interaction_terms, 'r3')

	def gradient_alpha12(self, R, interaction_terms):
		return self.two_step_forward_gradient(R, interaction_terms, 'alpha12')

	def gradient_alpha13(self, R, interaction_terms):
		return self.two_step_forward_gradient(R, interaction_terms, 'alpha13')

	def gradient_alpha21(self, R, interaction_terms):
		return self.two_step_forward_gradient(R, interaction_terms, 'alpha21')

	def gradient_alpha23(self, R, interaction_terms):
		return self.two_step_forward_gradient(R, interaction_terms, 'alpha23')

	def gradient_alpha31(self, R, interaction_terms):
		return self.two_step_forward_gradient(R, interaction_terms, 'alpha31')

	def gradient_alpha32(self, R, interaction_terms):
		return self.two_step_forward_gradient(R, interaction_terms, 'alpha32')


# ----- wrapper -------

	def quantity_interest(self,paras):
		R = paras[:3]

		# Inputs
		values = np.array(paras[3:])
		diagonal = self.self_interacting_terms

		# Build the 3x3 matrix
		interaction_terms = np.empty((3, 3), dtype=object)

		# Fill in the diagonal
		np.fill_diagonal(interaction_terms, diagonal)

		# Fill in the off-diagonal values row by row
		off_diag_indices = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
		for idx, (i, j) in enumerate(off_diag_indices):
			interaction_terms[i, j] = values[idx]

		return self.function_z(R,interaction_terms)[-1][-1]

	def Gradients(self,paras):
		R = paras[:3]

		# Inputs
		values = np.array(paras[3:])
		diagonal = self.self_interacting_terms

		# Build the 3x3 matrix
		interaction_terms = np.empty((3, 3), dtype=object)

		# Fill in the diagonal
		np.fill_diagonal(interaction_terms, diagonal)

		# Fill in the off-diagonal values row by row
		off_diag_indices = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
		for idx, (i, j) in enumerate(off_diag_indices):
			interaction_terms[i, j] = values[idx]

		self.df = self.function_z(R,interaction_terms)
		return np.array([self.gradient_r1(R,interaction_terms),self.gradient_r2(R,interaction_terms), self.gradient_r3(R,interaction_terms),
						self.gradient_alpha12(R,interaction_terms),self.gradient_alpha13(R,interaction_terms),self.gradient_alpha21(R,interaction_terms),self.gradient_alpha23(R,interaction_terms),self.gradient_alpha31(R,interaction_terms),self.gradient_alpha32(R,interaction_terms) ])
	def dateGenerating(self,sample_method ='Uniform'):
		if sample_method =='Uniform':
			pass
		elif sample_method =='POF-darts':
			pass
		else:
			raise ValueError(f"Expected Uniform/POF-darts, but got {sample_method}")


	def cross_section(self):
		# Function and parameter setup
		lotka = lotkaVolterra()
		function = lotka.quantity_interest
		domains = np.array([
			[0.1, 2],
			[0.1, 2],
			[0.1, 2],
			[0.25, 0.75],
			[0.25, 0.75],
			[0.25, 0.75],
			[0.25, 0.75],
			[0.25, 0.75],
			[0.25, 0.75]
		])
		para_default = np.array([0.5, 0.6, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

		# Mesh resolution
		n_points = 10

		fig, axes = plt.subplots(8, 8, figsize=(10, 10))
		plt.subplots_adjust(wspace=0.4, hspace=0.4)
		var_names = ['r1', 'r2', 'r3', 'α12', 'α13', 'α21', 'α23', 'α31', 'α32']


		for i in range(1,9):
			for j in range(8):
				ax = axes[i-1, j]

				if i <= j:
					ax.axis('off')
					continue

				# Create grid over variable i and j
				x_range = np.linspace(domains[i][0], domains[i][1], n_points)
				y_range = np.linspace(domains[j][0], domains[j][1], n_points)
				X, Y = np.meshgrid(x_range, y_range)
				Z = np.zeros_like(X)

				# Evaluate function over the grid
				for m in range(n_points):
					for n in range(n_points):
						para = para_default.copy()
						para[i] = X[m, n]
						para[j] = Y[m, n]
						Z[m, n] = function(para)

				# Plot contour
				contour = ax.contourf(X, Y, Z, levels=15, cmap='viridis')
				#ax.set_title(f'{var_names[i]} vs {var_names[j]}', fontsize=8)
				ax.tick_params(labelsize=6)
				if j == 0:
					ax.set_ylabel(var_names[i], fontsize=7)
				if i == 8:
					ax.set_xlabel(var_names[j], fontsize=7)

		# Add a single colorbar
		cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
		fig.colorbar(contour, cax=cbar_ax)



		#plt.suptitle("9×9 Cross-Section Contour Plots of Lotka–Volterra Model", fontsize=16)
		filename = f'Plots/lotka/cross_section.pdf'
		fig.savefig(filename, bbox_inches='tight', format = 'pdf')



def visualize_f_2d(var_dims, resolution=50, Gradients = False):
	"""
	Visualize f(a, b) as a 2D contour plot over selected components of a and b.

	Parameters:
	f          : function f(a, b)
	a_default  : default list of a parameters (e.g., [a1, a2, a3])
	b_default  : default list of b parameters (e.g., [b1, b2, b3])
	a_index    : index of a parameter to vary (e.g., 0 for a[0])
	b_index    : index of b parameter to vary (e.g., 1 for b[1])
	a_range    : (min, max) range for the selected a parameter
	b_range    : (min, max) range for the selected b parameter
	resolution : number of grid points per axis
	"""
	lotka = lotkaVolterra()
	domains = np.array([
		[0.1, 2],
		[0.1, 2],
		[0.1, 2],
		[0.25, 0.75],
		[0.25, 0.75],
		[0.25, 0.75],
		[0.25, 0.75],
		[0.25, 0.75],
		[0.25, 0.75]
	])
	para_default = [0.5,0.6,0.7,0.5,0.5,0.5,0.5,0.5,0.5]
	a_vals = np.linspace(*domains[var_dims[0]], resolution)
	b_vals = np.linspace(*domains[var_dims[1]], resolution)
	A, B = np.meshgrid(a_vals, b_vals)
	Z = np.zeros_like(A)

	for i in range(resolution):
		print(i)
		for j in range(resolution):
			para = para_default.copy()
			para[var_dims[0]] = A[i, j]
			para[var_dims[1]] = B[i, j]
			Z[i, j] = lotka.quantity_interest(para)
	lower_bounds = np.array(domains)[:, 0]
	upper_bounds = np.array(domains)[:, 1]


	plt.figure(figsize=(6, 5))
	contour = plt.contourf(A, B, Z, levels=50, cmap='viridis')
	plt.colorbar(contour)
	if Gradients == True:
		for i in range(20):
			sample1 = np.random.uniform(*domains[var_dims[0]])
			sample2 = np.random.uniform(*domains[var_dims[1]])

			# Construct full 6D input
			para = para_default.copy()
			para[var_dims[0]] = sample1
			para[var_dims[1]] = sample2

			# Get gradient and normalize
			grad = lotka.Gradients(para)
			g0, g1 = grad[var_dims[0]], grad[var_dims[1]]
			norm = np.sqrt(g0**2 + g1**2) + 1e-8  # Add epsilon to avoid zero division
			U = -0.1 * g0 / norm  # normalized to length 0.1
			V = -0.1 * g1 / norm

			plt.quiver(sample1, sample2, U, V, color='red', angles='xy', scale_units='xy', scale=1)
			print(f"Arrow at ({sample1:.2f}, {sample2:.2f}) with vector ({U:.3f}, {V:.3f})")
	plt.xlabel(f'var[{var_dims[0]}]')
	plt.ylabel(f'var[{var_dims[1]}]')
	plt.title('Contour plot of z3[10]')
	plt.tight_layout()
	plt.show()



def Gradient_check(h = 0.001):
	lotka = lotkaVolterra(method = 'Heun')
	alpha = np.array([
		[0.65, 0.6, 0.7],
		[0.3, 0.55, 0.4],
		[0.4, 0.5, 0.45]
	])
	for i in range(3):
		for j in range(3):
			if i != j:
				alpha_new = alpha.copy()
				alpha_new[i,j] = alpha_new[i,j] + h
				grad = (lotka.function_z([0.5,0.6,0.7], alpha)[-1][-1] - lotka.function_z([0.5,0.6,0.7], alpha_new)[-1][-1])/h
				g = getattr(lotka, f"gradient_alpha{i+1}{j+1}")([0.5,0.6,0.7],alpha)
				print(f"Gradient at alpha {i+1,j+1} is estimated as {grad}, with {g} predicted by class.")



# lotka = lotkaVolterra()
# lotka.cross_section()
#Gradient_check(h = 0.01)









