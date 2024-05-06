import numpy as np
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import solve
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
from sklearn.neighbors import KNeighborsRegressor
import random
import scipy.stats as stats
from POFdarts import POFdarts


#----------------------------------------- Brusselator Example -----------------------------------------
CONST_threshold = 3.75
Kmeans_Const = 1

##solve the model
def function_y(lambda1, lambda2,lambda3,T,n):
	""" 
	T: constant
	n: number of line spaces used to evaluate the intergral

	"""
	ls = np.zeros((n,2))
	ls[0] = (lambda3,1.0)
	t = T/n
	# loop from t to T-t
	for i in range(1,n):        
		def equations(vars):
			y1,y2 = vars
			eq1 = t*(lambda1 + y1**2*y2 - lambda2*y1 - y1) + ls[i-1][0] - y1
			eq2 = t*(lambda2*y1 - y1**2*y2) + ls[i-1][1] - y2
			return [eq1, eq2]
		ls[i] = fsolve(equations, ls[i-1])
	return ls

####Integral approximation
def integral(lambda1, lambda2,lambda3,T= 5,n = 100):
	#Riemann sum approxiamation left sided
	return (1/T)*np.sum(np.sum(function_y(lambda1, lambda2,lambda3,T,n),axis = 1)*T/n )

def integrals(lambda1, lambda3 = 1.65,T = 5,n = 100):
		#Riemann sum approxiamation left sided
		return (1/T)*np.sum(np.sum(function_y(lambda1[0], lambda1[1],lambda3,T,n),axis = 1)*T/n )

def integral_3D(lambda_,T= 5,n = 100):
	return (1/T)*np.sum(np.sum(function_y(lambda_[0], lambda_[1],lambda_[2],T,n),axis = 1)*T/n )


## partial derivatives
def dy_dlambda1_t(lambda1, lambda2,lambda3,T,n):
	ls = np.zeros((n,2))
	ls[0] = (0,0)
	t = 0
	F = function_y(lambda1, lambda2,lambda3, T,n)
	for i in range(1,n):
		t += T/n
		Y = F[i]
		def equations(vars):
			y1,y2 = vars
			eq1=  T/n * (1 + 2*Y[0]*Y[1]*y1 + Y[0]**2*y2 - lambda2*y1 - y1)  + ls[i-1][0] - y1
			eq2=  T/n * (lambda2*y1 - 2*Y[0]*Y[1]*y1 - Y[0]**2*y2) + ls[i-1][1] - y2
			return [eq1, eq2]
		ls[i] = fsolve(equations, ls[i-1])
	return ls

def dy_dlambda2_t(lambda1, lambda2,lambda3,T,n):
	ls = np.zeros((n,2))
	ls[0] = (0,0)
	t = 0
	F = function_y(lambda1, lambda2,lambda3, T,n)
	for i in range(1,n):
		t += T/n
		Y = F[i]
		def equations(vars):
			y1,y2 = vars
			eq1=  T/n * (2*Y[0]*Y[1]*y1 + Y[0]**2*y2 - Y[0]-lambda2*y1 - y1)  + ls[i-1][0] - y1
			eq2=  T/n * (Y[0] + lambda2*y1 - 2*Y[0]*Y[1]*y1 - Y[0]**2*y2) + ls[i-1][1] - y2
			return [eq1, eq2]
		ls[i] = fsolve(equations, ls[i-1])
	return ls

def dy_dlambda3_t(lambda1, lambda2,lambda3,T,n):
	ls = np.zeros((n,2))
	ls[0] = (1,0)
	t = 0
	F = function_y(lambda1, lambda2,lambda3, T,n)
	for i in range(1,n):
		t += T/n
		Y = F[i]
		def equations(vars):
			y1,y2 = vars
			eq1=  T/n * (2*Y[0]*Y[1]*y1 + Y[0]**2*y2 -lambda2*y1 - y1)  + ls[i-1][0] - y1
			eq2=  T/n * (lambda2*y1 - 2*Y[0]*Y[1]*y1 - Y[0]**2*y2) + ls[i-1][1] - y2
			return [eq1, eq2]
		ls[i] = fsolve(equations, ls[i-1])
	return ls


#Gradient 
def dQ_dlambda(lambda1, lambda2,lambda3 = 1.65,T = 5,n = 100):
	sum1 = (1/T)*np.sum(np.sum(dy_dlambda1_t(lambda1, lambda2,lambda3,T,n),axis = 1)*T/n )
	sum2 = (1/T)*np.sum(np.sum(dy_dlambda2_t(lambda1, lambda2,lambda3,T,n),axis = 1)*T/n )
	return np.array([sum1,sum2])


def DQ_Dlambda(Lambda,lambda3 = 1.65,T = 5,n = 100):
	sum1 = (1/T)*np.sum(np.sum(dy_dlambda1_t(Lambda[0], Lambda[1],lambda3,T,n),axis = 1)*T/n )
	sum2 = (1/T)*np.sum(np.sum(dy_dlambda2_t(Lambda[0], Lambda[1],lambda3,T,n),axis = 1)*T/n )
	return np.array([sum1,sum2])

def DQ_Dlambda_3D(Lambda,T = 5,n = 100):
	sum1 = (1/T)*np.sum(np.sum(dy_dlambda1_t(Lambda[0], Lambda[1],Lambda[2],T,n),axis = 1)*T/n )
	sum2 = (1/T)*np.sum(np.sum(dy_dlambda2_t(Lambda[0], Lambda[1],Lambda[2],T,n),axis = 1)*T/n )
	sum3 = (1/T)*np.sum(np.sum(dy_dlambda3_t(Lambda[0], Lambda[1],Lambda[2],T,n),axis = 1)*T/n )
	return np.array([sum1,sum2, sum3])



#----------------------------------------- Elliptical -----------------------------------------

# gradient part imcomplete

class elliptic(object):
	def __init__(self, n):
		self.df = []
		self.u = []
		self.n = n
		self.du_dlambda_1  = []
		self.du_dlambda_2  = []
		self.du_dlambda_3  = []
		self.du_dlambda_4  = []

	def function_y(self, Lambda):
		'''
		'''
		n = self.n
		x_seq = np.arange(0,n+1)*1/n
		A = np.zeros((n + 1, n + 1))
		C = np.zeros(n + 1)

		for i in range(n+1):
			if i == 0:
				A[i,0] = 1
				C[i] = 0
			elif i == n:
				A[i,i] = 1
				C[i] = 0
			else:
				A[i, i-1 : i +2] =  self.coefficients(Lambda,x_seq[i])
				x = x_seq[i]
				C[i] = (1 - x)*np.tanh(4*(x - Lambda[2])) + np.sin(5*np.pi*Lambda[3]*x)

		# A*x = C
		self.u = solve(A, C)
		return self.u

	def mapping_Q(self,type_ = "mean"):
		'''
		'''

		if type_ == "mean":
			return np.sum(self.u*(1/self.n))


	def dQ_dlambda_1(self,Lambda):

		# check self.Lambda == Lambda
		n = self.n
		x_seq = np.arange(0,n+1)*1/n
		A = np.zeros((n + 1, n + 1))
		C = np.zeros(n + 1)

		for i in range(n+1):
			if i == 0:
				A[i,0] = 1
				C[i] = 0
			elif i == n:
				A[i,i] = 1
				C[i] = 0
			else:
				A[i, i-1 : i +2] =  self.coefficients(Lambda,x_seq[i])
				x = x_seq[i]
				C[i] =  - np.sum([k*j for k, j in zip(self.coefficients_dlambda_1(Lambda,x_seq[i]), self.u[i-1 : i +2])])

		# A*x = C
		self.du_dlambda_1 = solve(A, C)

	def dQ_dlambda_2(self,Lambda):

		# check self.Lambda == Lambda
		n = self.n
		x_seq = np.arange(0,n+1)*1/n
		A = np.zeros((n + 1, n + 1))
		C = np.zeros(n + 1)

		for i in range(n+1):
			if i == 0:
				A[i,0] = 1
				C[i] = 0
			elif i == n:
				A[i,i] = 1
				C[i] = 0
			else:
				A[i, i-1 : i +2] =  self.coefficients(Lambda,x_seq[i])
				x = x_seq[i]
				C[i] =  - np.sum([k*j for k, j in zip(self.coefficients_dlambda_2(Lambda,x_seq[i]), self.u[i-1 : i +2])])

		# A*x = C
		self.du_dlambda_2 = solve(A, C)

	def dQ_dlambda_3(self,Lambda):

		# check self.Lambda == Lambda
		n = self.n
		x_seq = np.arange(0,n+1)*1/n
		A = np.zeros((n + 1, n + 1))
		C = np.zeros(n + 1)

		for i in range(n+1):
			if i == 0:
				A[i,0] = 1
				C[i] = 0
			elif i == n:
				A[i,i] = 1
				C[i] = 0
			else:
				A[i, i-1 : i +2] =  self.coefficients(Lambda,x_seq[i])
				x = x_seq[i]
				C[i] =  (1 - x)*(1 - np.tanh(4*(x - Lambda[2]))**2)*(-4)

		# A*x = C
		self.du_dlambda_3 = solve(A, C)

	def dQ_dlambda_4(self,Lambda):

		# check self.Lambda == Lambda
		n = self.n
		x_seq = np.arange(0,n+1)*1/n
		A = np.zeros((n + 1, n + 1))
		C = np.zeros(n + 1)

		for i in range(n+1):
			if i == 0:
				A[i,0] = 1
				C[i] = 0
			elif i == n:
				A[i,i] = 1
				C[i] = 0
			else:
				A[i, i-1 : i +2] =  self.coefficients(Lambda,x_seq[i])
				x = x_seq[i]
				C[i] =  np.cos(5*np.pi*Lambda[3]*x)*(5*np.pi*x)

		# A*x = C
		self.du_dlambda_4 = solve(A, C)



	def Gradient_Q(self,Lambda, type_ = "mean"):
		'''
		'''
		self.function_y(Lambda)
		self.dQ_dlambda_1(Lambda)
		self.dQ_dlambda_2(Lambda)
		self.dQ_dlambda_3(Lambda)
		self.dQ_dlambda_4(Lambda)

		if type_ == "mean":
			return([np.sum(self.du_dlambda_1*(1/self.n)), np.sum(self.du_dlambda_2*(1/self.n)), np.sum(self.du_dlambda_3*(1/self.n)), np.sum(self.du_dlambda_4*(1/self.n))] )

	def coefficients(self,Lambda, x):
		'''
		return coefficient for u(x-h), u(x), u(x+h)
		'''
		h = 1/self.n
		a = -(2*x*np.exp(-Lambda[0]*x) - Lambda[0]*x**2*np.exp(-Lambda[0]*x) - Lambda[1])
		b = (x**2*np.exp( -Lambda[0]*x ) + 0.05 )

		return [ b/h**2 - a/h, a/h - 2*b/h**2, b/h**2]

	def coefficients_dlambda_1(self,Lambda,x):
		'''
		return coefficient for u(x-h), u(x), u(x+h) as defined in appendex B
		'''
		h = 1/self.n
		a = -(2*x*np.exp(-Lambda[0]*x) - Lambda[0]*x**2*np.exp(-Lambda[0]*x) - Lambda[1])
		b = (x**2*np.exp( -Lambda[0]*x ) + 0.05 )
		da_dlambda_1 = 3*x**2*np.exp(-Lambda[0]*x) - Lambda[0]*x**3*np.exp(-Lambda[0]*x)
		db_dlambda_1 = -x**3*np.exp(-Lambda[0]*x)

		return [ db_dlambda_1/h**2 - da_dlambda_1/h, da_dlambda_1 /h - 2*db_dlambda_1/h**2, db_dlambda_1/h**2]

	def coefficients_dlambda_2(self,Lambda,x):
		h = 1/self.n
		da_dlambda_2 = 1
		db_dlambda_2 = 0
		return [ db_dlambda_2/h**2 - da_dlambda_2/h, da_dlambda_2 /h - 2*db_dlambda_2/h**2, db_dlambda_2/h**2]

	# lamda 3/4 has 0 coefficients


def elliptic_function(Lambda, n = 100):
	elliptic_gen = elliptic(n)
	elliptic_gen.function_y(Lambda)
	Q = elliptic_gen.mapping_Q()
	del elliptic_gen
	return Q

def elliptic_Gradient(Lambda, n = 100):
	elliptic_gen = elliptic(n)
	dQ = elliptic_gen.Gradient_Q(Lambda)
	del elliptic_gen
	return dQ







# Simulated example 1 -------------------------
def function1(x):
	return ((x[1]-.5*(np.tanh(20*x[0])*np.tanh(20*(x[0]-.5))+1)*np.exp(.2*x[0]**2)))**2

def Gradient_f1(X):
	x = X[0]
	y = X[1]
	dx = 2*((y-.5*(np.tanh(20*x)*np.tanh(20*(x-.5))+1)*np.exp(.2*x**2)))*( 
		-0.5*(
				(0.4*x)*np.exp(0.2*x**2)*(np.tanh(20*x)*np.tanh(20*(x-.5)) + 1 )+
				np.exp(0.2*x**2)*(( 1 - np.tanh(20*x)**2 )*20*np.tanh(20*(x-.5))+ 
									np.tanh(20*x)*(1 - np.tanh(20*(x-.5))**2)*20)
		)
	)
	dy = 2*((y-.5*(np.tanh(20*x)*np.tanh(20*(x-.5))+1)*np.exp(.2*x**2)))
	return np.array([dx,dy])



# Simulated example 2 -------------------------

def function2(X, A = 10, B = 1):
	x = X[0]
	y = X[1]
	return 1 + np.tanh(B*(y - A*x*(x - 1/2)*(x+1/2)))

def Gradient_f2(X, A = 10, B = 1):
	x = X[0]
	y = X[1]
	dx = B*(1 - np.tanh(B*(-A*x*(x - 0.5)*(x + 0.5) + y))**2)*(-A*x*(x - 0.5) - A*x*(x + 0.5) - A*(x - 0.5)*(x + 0.5))
	dy = B*(1 - np.tanh(B*(-A*x*(x - 0.5)*(x + 0.5) + y))**2)
	return np.array([dx,dy]) 






# comparison data set for PSVM and GPSVM
#--------------------------------------------------------Balanced data----------------------------------------
def create_binary_class_boundary_twoSquares(num_points):
	# Generate random 2D binary class data points using NumPy
	x_values = np.random.uniform(0, 10, num_points)
	y_values = np.random.uniform(0, 10, num_points)
	
	# Create a decision boundary consisting of two connected line segments
	x_boundary = [1, 1, 4, 4, 1]
	y_boundary = [1, 4, 4, 1, 1]
	
	# Determine the labels based on whether points are inside or outside the boundary
	labels = []
	for x, y in zip(x_values, y_values):
		if ( x <= 5):
			label = -1  # Outside the boundary
		else:
			label = 1  # Inside the boundary
		labels.append(label)

	# Create a DataFrame
	data = {
		'X': x_values,
		'Y': y_values,
		'Label': labels
	}
	df = pd.DataFrame(data)
	return df[(df['X'] >= 5) | (df['X'] <= 5)]



def create_binary_class_boundary_twoCircles(num_points):
	# Generate random 2D binary class data points using NumPy
	x_values = np.random.uniform(0, 10, num_points)
	y_values = np.random.uniform(0, 10, num_points)
	
	
	# Determine the labels based on whether points are inside or outside the boundary
	labels = []
	for x, y in zip(x_values, y_values):
		if ( (x - 5 )**2 + (y-5)**2 <= 6):
			label = -1  # Outside the boundary
		else:
			label = 1  # Inside the boundary
		labels.append(label)

	# Create a DataFrame
	data = {
		'X': x_values,
		'Y': y_values,
		'Label': labels
	}
	df = pd.DataFrame(data)
	return df




def create_binary_class_boundary_concave(num_points):
	# Generate random 2D binary class data points using NumPy
	x_values = np.random.uniform(0, 10, num_points)
	y_values = np.random.uniform(0, 10, num_points)
	
	
	# Determine the labels based on whether points are inside or outside the boundary
	labels = []
	for x, y in zip(x_values, y_values):
		if ( x <= 2 or x >= 8 or y >= 8  ):
			label = 1  
		elif ( x >= 4 and x <= 6 and y >= 2):
			label = 1  
		else:
			label = -1
		labels.append(label)

	# Create a DataFrame
	data = {
		'X': x_values,
		'Y': y_values,
		'Label': labels
	}
	df = pd.DataFrame(data)
	return df



def create_binary_class_boundary_spiral(num_points):
	n1 = int(num_points/2)
	n2 = num_points - n1
	theta = np.random.uniform(0, 2*np.pi, n1)

	a = 0.75
	sd = 3.5
	alpha = 1.75

	x = (1+ theta)**a * np.cos(theta) + (np.random.beta(alpha, alpha,n1) - 0.5)*sd/2
	y = (1+ theta)**a * np.sin(theta) - 0.75 + (np.random.beta(alpha, alpha,n1) - 0.5)*sd/2 
	l1 = np.repeat(-1, n1)

	theta2 = np.random.uniform(0, 2*np.pi, n2)
	x2 = -(1+ theta2)**a * np.cos(theta2) + (np.random.beta(alpha, alpha,n2) - 0.5)*sd/2
	y2 = -(1+ theta2)**a * np.sin(theta2) + 0.75 + (np.random.beta(alpha, alpha,n2) - 0.5)*sd/2  -1
	l2 = np.repeat(1, n2)

	data = {
		'X': np.concatenate((x, x2)),
		'Y': np.concatenate((y, y2)),
		'Label': np.concatenate((l1, l2))
	}
	df = pd.DataFrame(data)
	return df

#------------------------------------------------------------------------------------------------------
#----------------------------------------- Data_Generation---------------------------------------------

class SIP_Data(object):
	'''
	function_y and function_gradient takes 1 vector argument 
	'''
	def __init__(self, function_y, function_gradient,CONST_threshold, dim, *domain):
		if len(domain) != dim:
			raise ValueError(f"Expected {dim} domain intervals, but got {len(domain)}")
		
		# Check if all arguments are tuples with two elements
		for arg in domain:
			if len(arg) != 2:
				raise TypeError("Each domain argument must be a tuple of two numbers (low, high)")

		self.CONST_threshold = CONST_threshold
		self.dim = dim
		self.domain = domain
		self.function_y = function_y
		self.function_gradient = function_gradient
		self.df = []
		self.Gradient = []

	def generate_Uniform(self,n, Initialization = True):
		points = np.array([np.random.uniform(low, high, n) for low, high in self.domain]).T

		self.df = pd.DataFrame(points, columns = [f'X{i+1}' for i in range(self.dim)])

		Z = self.df.apply(self.function_y, axis = 1)
		self.Gradient = self.df.apply(self.function_gradient, axis = 1)
		self.df['Label'] = np.zeros(n) -1
		index = Z >= self.CONST_threshold 
		self.df['Label'][index] = 1
		self.df['f'] = Z
		

	def generate_POF(self,n,CONST_a ,iniPoints = 10, max_iterations  = 1000, max_miss = 1000,sampleCriteria = 'k-dDarts', Initialization = True, adaptive = False, adaptiveRatio = 2, shrink_ratio = None):
		if Initialization == True:
			self.POFdarts = POFdarts( self.function_y, self.function_gradient , CONST_a,  self.CONST_threshold , max_iterations  = max_iterations, max_miss = max_miss, adaptive = adaptive,
				adaptiveRatio = adaptiveRatio, shrink_ratio = shrink_ratio)
			self.POFdarts.Initialize(iniPoints, self.dim , self.domain)
		self.POFdarts.Generate_data( n - iniPoints, self.dim , self.domain, sampleCriteria = sampleCriteria)
		self.Gradient = self.POFdarts.Q

		#self.df = pd.DataFrame(data={"X":np.array(self.POFdarts.df)[:,0],"Y":np.array(self.POFdarts.df)[:,1]})
		self.df = pd.DataFrame(np.array(self.POFdarts.df), columns = [f'X{i+1}' for i in range(self.dim)])

		self.df['Label'] = np.zeros( len(self.POFdarts.df) ) -1
		index = np.array(self.POFdarts.y) >= self.CONST_threshold 
		self.df['Label'][index] = 1
		self.df['f'] = self.POFdarts.y














