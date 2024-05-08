import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
import random
import statistics

# ---------------------------------- MagKmeans Algorithm ---------------------------------- 
# Code based on the pseudocode from the paper:
# 	Title: "POF-Darts: Geometric Adaptive Sampling for Probability of Failure"
# 	Authors: Ebeida, Mohamed and Mitchell, Scott and Swiler, Laura and Romero, Vicente and Rushdi, Ahmad
# 	Published: Reliability Engineering & System Safety, vol 155, 2016
# Class POFdarts implementation adapted from the paper's pseudocode.


class POFdarts(object):
	def __init__(self, function_y, gradient, CONST_a,  critical_values, lower_bound = 0.01,max_iterations  = 50000, max_miss = 10000,random_state = 0):
		'''
		function_y takes 1 argument: tuple
		gradient takes 1 argument: tuple
		'''
		self.function_y = function_y
		self.lower_bound = 0.01
		self.gradient = gradient
		self.CONST_a = CONST_a
		# critical_values is a list 
		if isinstance(critical_values, (list, np.ndarray)):
			self.critical_values = [float(c) for c in critical_values]
		else:
			raise ValueError("Input must be a list or a numpy array.")

		self.max_iterations = max_iterations
		self.max_miss = max_miss
		self.max_d = 0.0
		self.seed = random_state
		# data frame is a list object
		self.df = []
		self.radius = []
		self.y = []
		self.Q = []


	def contain(self,point):
		for i in range(len(self.df) ):
			if np.linalg.norm(np.array(point) - np.array(self.df[i]) ) <= self.radius[i]:
				return True
		return False

	def remove_overlap(self):
		# complete overlap
		for critical_value in self.critical_values:			
			index = np.array(self.y) < critical_value
			I = [i for i, x in enumerate(index) if x]
			indexInv = [not i for i in index]
			J= [i for i, x in enumerate(index) if not x]
			for i in I:
				for j in J:
					if np.linalg.norm( np.array(self.df[i]) - np.array(self.df[j]) ) < self.radius[i] + self.radius[j]:
						L = np.abs( np.array(self.y[i]) -  np.array(self.y[j]) )/np.linalg.norm(np.array(self.df[i]) - np.array(self.df[j]) )
						self.radius[i] =  min(self.radius[i],  np.abs( np.array(self.y[i]) - critical_value)/L)
						self.radius[j] =  min(self.radius[j],np.abs( np.array(self.y[j]) - critical_value)/L)
		return

	def addpoint(self, newPoint):
		z = self.function_y(np.array(newPoint))
		self.y.append(z)
		Q = self.gradient(newPoint)
		self.Q.append(Q)
		# 
		r = np.min(np.abs(z - np.array(self.critical_values))) /( self.CONST_a*np.max([np.linalg.norm(Q),self.lower_bound]))
		self.radius.append(r)
		self.df.append(newPoint)
		self.remove_overlap()

	def update_radius(self):
		self.radius = [r * 2/3 for r in self.radius]



	def Initialize(self, iniPoints,  dim, args):
		random.seed(self.seed)
		# Check if the domain for each dimension is provided
		if len(args) != dim:
			raise ValueError(f"Expected {dim} domain intervals, but got {len(args)}")
		
		# Check if all arguments are tuples with two elements
		for arg in args:
			if len(arg) != 2:
				raise TypeError("Each domain argument must be a tuple of two numbers (low, high)")

		points = np.array([np.random.uniform(low, high, iniPoints) for low, high in args]).T.tolist()


		self.df = points 
		self.y = pd.DataFrame(self.df).apply(self.function_y, axis = 1).values.tolist()
		# calculate gradient
		self.radius= np.zeros(iniPoints)
		for i in range(iniPoints):
			Q = self.gradient(self.df[i])
			self.Q.append(Q)
			self.radius[i]= np.min(np.abs(self.y[i] - np.array(self.critical_values))) /( self.CONST_a *np.linalg.norm(Q) )
		self.radius = self.radius.tolist()
		self.remove_overlap()

	def Generate_data(self, N, dim, args, sampleCriteria = 'k-dDarts'):
		'''
		sample method: k-dDarts, accept-reject
		'''
		i = 0
		if sampleCriteria == 'accept-reject':
			while i < N:
				newPoint = np.array([np.random.uniform(low, high, 1) for low, high in args]).T.tolist()[0]

				counter = 0

				while self.contain(newPoint) and counter < self.max_iterations:
					newPoint = np.array([np.random.uniform(low, high, 1) for low, high in args]).T.tolist()[0]
					counter = counter + 1
				# no points found to add
				if counter == self.max_iterations:
					print('decrease')
					self.CONST_a = self.CONST_a * 3/2
					self.update_radius()
				# else found point to add
				else:
					self.addpoint(newPoint)
					i = i+1
		else:
			while i < N:
				newPoint = self.lineDartSample(dim,args)
				if newPoint is None:
					print('missed, decrease')
					print(self.radius)
					self.CONST_a = self.CONST_a * 3/2
					self.update_radius()
				else:
					newPoint = newPoint.tolist()
					self.addpoint(newPoint)
					i = i+1
		return 0




	def lineDartSample(self, dim, args):
		totalmiss = 0
		while totalmiss < self.max_miss:
			linearDart = np.array([np.random.uniform(low, high, 1) for low, high in args]).T[0]
			dartsDim = np.arange(len(linearDart))
			random.shuffle(dartsDim)
			for i in dartsDim:
				# g = [a0b0a1b2....bq]
				g_lineSegment = np.array([float(k) for k in args[i] ])
				# remove intersections from g.
				for diskCenter, radius,k in zip(self.df, self.radius, range(len(self.df)) ):
					if len(g_lineSegment) ==0:
						break
					minDist = np.sqrt(np.linalg.norm(diskCenter - linearDart)**2 - (linearDart[i] - diskCenter[i])**2)
					if minDist < 0:
						raise ValueError("distance must be non-nagetive.")
					if minDist >= radius:
						continue
					else: # overlap
						# find the intersections [b_, a_]:
						seg = np.sqrt(radius**2 - minDist**2)
						b_ = diskCenter[i] - seg
						a_ = diskCenter[i] + seg

						# ------------------ remove the overlap ------------------ 
						#--Case 1: [b_, a_] are beyond a0 or bq, then does not intersect
						indexa_ = find_index(g_lineSegment, a_)
						indexb_ = find_index(g_lineSegment, b_)
						if b_ >= g_lineSegment[-1] or a_ <= g_lineSegment[0]:
							continue
						#--Case 2: [b_, a_]  lie between a_j abd b_j, split [a_j, b_j] into [a_j,b_,a_,b_j]
						elif indexa_ == indexb_ and indexa_ != -1 and (indexb_ % 2) == 0:
							g_lineSegment = np.concatenate([ g_lineSegment[0:indexa_+1],[b_, a_], g_lineSegment[indexa_+1:] ])
						#--Case 3: 
						else:
							#only b_ lies between a_j and b_j, let  b_j = b_
							if indexb_ != -1 and (indexb_ % 2) == 0:
								g_lineSegment[indexb_ +1]  = b_
							#only a_ lies between a_j and b_j, let a_j = a_
							if indexa_ != -1 and (indexa_ % 2) == 0:
								g_lineSegment[indexa_]  = a_
							# remove any line segment in-between [b_, a_]
							g_lineSegment = g_lineSegment[ np.logical_or(g_lineSegment <= b_ , g_lineSegment >= a_)]
				# check if g is empty:
				if len(g_lineSegment) ==0:
					#print('miss at X',i)
					continue
				else:
					# sample from g
					lengthArr = np.zeros( int(len(g_lineSegment)/2) )
					for j in range( len(lengthArr)):
						# length = b_j - a_j
						lengthArr[j] =  g_lineSegment[j*2 + 1]  - g_lineSegment[j*2]
					sample = random.uniform(0, np.sum(lengthArr))
					total = 0
					point_i = 0
					for j in range( len(lengthArr)):
						total +=  lengthArr[j]
						if total >= sample:
							point_i = g_lineSegment[j*2] + sample - total +  lengthArr[j]
							break

					linearDart[i] = point_i
					return(linearDart)
			# print('miss')
			totalmiss += 1
		return 0








def find_index(A, constant):
	low, high = 0, len(A) - 1

	while low <= high:
		mid = (low + high) // 2

		# Check if mid is the last index or if the array is not large enough
		if mid == len(A) - 1:
			return -1

		# Check if A[mid] < constant and A[mid + 1] > constant
		if A[mid] < constant and A[mid + 1] > constant:
			return mid

		# Adjust search range
		if A[mid] >= constant:
			high = mid - 1
		else:
			low = mid + 1
	return -1  # Return -1 if no index is found











