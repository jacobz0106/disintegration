# MagKmeans algorithm
import numpy as np
import pandas as pd
import random
import time
import cvxpy as cp
from scipy.spatial import distance
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from dataGeneration import *
import scipy.stats as stats
from matplotlib import cm
from gurobipy import Model, GRB, quicksum
import gurobipy as gp



# Function used to find the distance matrix.
def calculate_distances(df1, df2, metric='euclidean'):
	# Ensure the two DataFrames have the same columns and order
	df2 = df2[df1.columns]

	# Convert DataFrames to NumPy arrays for efficient calculations
	arr1 = df1.to_numpy()
	arr2 = df2.to_numpy()

	# Calculate distances using the specified metric
	if metric == 'euclidean':
		dist_matrix = distance.cdist(arr1, arr2, 'euclidean')
	elif metric == 'manhattan':
		dist_matrix = distance.cdist(arr1, arr2, 'cityblock')
	# Add more distance metrics as needed

	# Create a DataFrame from the distance matrix
	df_c = pd.DataFrame(dist_matrix, index=df1.index, columns=df2.index)

	return df_c

def Euclidean_distance_vector(B, A_train):
	return np.array([np.linalg.norm(B - A_i) for A_i in A_train ]) 

# 
def convert_to_binary(Z):
	'''
	Convert a matrix Z into a binary matrix where each entry is 1 if it is the row maximum.

	Parameters:
	Z (numpy.ndarray): Input matrix where each row sums to 1.

	Returns:
	numpy.ndarray: Binary matrix where each entry is 1 if it is the row maximum.
	'''
	# Find the indices of the maximum values in each row
	max_indices = np.argmax(Z, axis=1)
	
	# Create a binary matrix with zeros
	binary_Z = np.zeros_like(Z)
	
	# Set the maximum values to 1 based on the max_indices
	binary_Z[np.arange(Z.shape[0]), max_indices] = 1
	
	return binary_Z





# ---------------------------------- MagKmeans Algorithm ---------------------------------- 
# Code based on the pseudocode from the paper:
# 	Title: "Efficient Algorithm for Localized Support Vector Machine"
# 	Authors: Cheng, Haibin and Tan, Pang-Ning and Jin, Rong
# 	Published: IEEE Transactions on Knowledge and Data Engineering, 2009, vol 2, number 4
# Class MagKmeans implementation adapted from the paper's pseudocode.
# Linear program is solved with cvxpy package.
class MagKmeans(object):
	def __init__(self, n_clusters, max_iterations  = 500, random_state = 0):
		#cluster membership matrix
		self.clusterMembership = []
		self.K = n_clusters
		self.constK  = n_clusters
		# ndarray of shape (n_clusters, n_features)
		self.cluster_centers_ = []
		# ndarray of shape (n_samples,)
		self.labels_ = []
		self.dfTrain = []
		self.dfLabel = []
		self.R = 0
		self.max_iterations = max_iterations
		self.random_state = random_state

	# Linear program solver
	def update_cluster_membership(self):
		"""
		Update cluster membership using linear programming to minimize within-cluster differences + penalty in class distribution.

		"""
		X = self.dfTrain
		C = self.cluster_centers_
		Y = self.dfLabel

		n, d = X.shape
		K = C.shape[0]

		# Create binary variables for cluster membership
		Z = cp.Variable((n, K))

		# Define the optimization objective
		objective = cp.Minimize(cp.sum(cp.norm(X - Z @ C, axis = 1)) + cp.sum(cp.abs(self.R *  cp.matmul(np.matrix(Y), Z)  )))


		
		# Define constraints
		constraints = [cp.sum(Z, axis=1) == 1, Z >= 0, Z <= 1]  # Each data point belongs to exactly one cluster

		# Create the optimization problem
		problem = cp.Problem(objective, constraints)

		# List of solvers to try
		solvers = [ "GUROBI", "ECOS", "SCS"]

		optimal_solution_found = False
		# Iterate through the solvers
		for solver in solvers:
			if solver == "GUROBI":
				# Create a new Gurobi model
				gp_env = gp.Env(empty=True) 
				#suppress or show output
				gp_env.setParam("OutputFlag",0)
				gp_env.start()
				m = gp.Model("gp model",env=gp_env)
				# Create binary variables for cluster membership
				#Z = m.addVars(n, K, vtype=GRB.BINARY, name="Z")
				Z = m.addVars(n, K, lb=0, ub=1, name="Z")
				

				# Squared differences part
				objective_norm_part = quicksum((X[i, j] - quicksum(Z[i, k] * C[k, j] for k in range(K)))**2 
											   for i in range(n) for j in range(d))

				# Initialize the absolute value part of the objective
				objective_abs_part = 0
				# Calculate absolute values part - this needs to be linearized
				for i in range(n):
					for k in range(K):
						abs_expr = self.R * Y[i] * Z[i, k]
						# Linearize the absolute value (requires introducing new variables)
						pos = m.addVar()
						neg = m.addVar()
						m.addConstr(pos >= abs_expr)
						m.addConstr(neg >= -abs_expr)
						objective_abs_part += pos + neg

				# Total objective
				objective = objective_norm_part + objective_abs_part

				# Set the objective in the model
				m.setObjective(objective, GRB.MINIMIZE)


				# Define constraints
				for i in range(n):
					m.addConstr(quicksum(Z[i, k] for k in range(K)) == 1)  # Each data point belongs to exactly one cluster

				# Optimize model
				m.optimize()
				# Retrieve the solution
				solution = np.zeros((n, K))
				if m.status == GRB.OPTIMAL:
					for i in range(n):
						for k in range(K):
							solution[i, k] = Z[i, k].X
					Z_optimal = solution
					optimal_solution_found = True
					m.dispose()
					gp_env.dispose()
					break
				else:
					print("No optimal solution found.")
					m.dispose()
					gp_env.dispose()

			else:#use cvxpy to solve
				try:
					# Solve the LP problem using the current solver
					problem.solve(solver=solver)

					# Check if the solver found an optimal solution
					if problem.status == cp.OPTIMAL:
						Z_optimal = Z.value
						optimal_solution_found = True
						break
					else:
						print(f"Solver {solver} did not find an optimal solution.")
				except Exception as e:
					print(f"Solver {solver} encountered an error: {e}")
		# Get the optimized cluster memberships

		# Z_optimal = Z.value
		if optimal_solution_found:
			self.clusterMembership = copy.deepcopy(convert_to_binary(Z_optimal))
		else:
			#raise Exception("Optimization problem not solved optimally.")
			print("Optimization problem not solved optimally.")
			self.clusterMembership = copy.deepcopy(convert_to_binary(Z_optimal))

	def update_cluster_centroids(self):
		"""
		Update cluster centroids based on the cluster membership matrix.

		Parameters:
		data (numpy.ndarray): Input data points, where each row represents a data point.
		cluster_membership (numpy.ndarray): Cluster membership matrix, where each row corresponds
										   to a data point and each column represents a cluster.
										   Each entry is 1 if the data point belongs to that cluster,
										   0 otherwise.

		Returns:
		numpy.ndarray: Updated cluster centroids.
		"""
		# Ensure that data and cluster_membership have the same number of rows
		if self.dfTrain.shape[0] != self.clusterMembership.shape[0]:
			raise ValueError("Data and cluster_membership must have the same number of rows.")
		
		# Get the number of clusters
		num_clusters = self.K
		
		# Initialize an empty list to store the updated centroids
		updated_centroids = []
		
		# Iterate over each cluster
		for cluster_idx in range(self.K):
			# Select data points that belong to the current cluster
			cluster_points = self.dfTrain[self.clusterMembership[:, cluster_idx] == 1]
			
			# Check if there are data points in the cluster
			if len(cluster_points) > 0:
				# Update the cluster centroid by taking the mean of data points in the cluster
				updated_centroid = np.mean(cluster_points, axis=0)
				updated_centroids.append(updated_centroid)
			else:
				# If there are no data points in the cluster, reduce the cluster number by 1
				num_clusters -= 1
				# print('One cluster removed.\n')
		# update parameters
		self.K = num_clusters
		# check termination
		# Convert centroids to a set representation to compare ignoring row order
		initial_centroids_set = set(map(tuple, self.cluster_centers_))
		previous_centroids_set = set(map(tuple, updated_centroids))
		# Check if the initial centroids are the same as the previous centroids
		if initial_centroids_set == previous_centroids_set:
			return 1
		else:
			self.cluster_centers_ = np.array(updated_centroids)
		return 0

	def initialize_k_cluster_centroids(self):
		"""
		Initialize K cluster centroids using random data points from the input data.

		Parameters:
		data (numpy.ndarray): Input data matrix, where each row represents a data point,
							 and each column corresponds to a feature.
		k (int): Number of clusters.

		Returns:
		numpy.ndarray: Matrix containing the initial K cluster centroids.
		"""
		# Check if the number of clusters (k) is valid

		if self.K <= 0 or self.K > self.dfTrain.shape[0]:
			raise ValueError("Invalid number of clusters (k).")
		
		# Randomly select K unique indices for data points

		centroid_indices = np.random.choice(self.dfTrain.shape[0], self.K, replace=False)
		
		# Initialize cluster centroids with the selected data points
		initial_centroids = self.dfTrain[centroid_indices, :]
		
		self.cluster_centers_ = initial_centroids
		return




	def fit(self, dfTrain, dfLabel, R):
		self.dfTrain = dfTrain
		self.R = R
		valid_values = {-1, 1}  # Use a set for faster membership checking

		# Check if the input dfLabel is a numpy array contaning -1 and 1
		if isinstance(dfLabel, (list, np.ndarray)):
			unique_values = set(dfLabel)
			if not (valid_values.issubset(unique_values) and len(unique_values) == 2):
				raise ValueError("Input array must contain both -1 and 1.")
		else:
			raise ValueError("Input must be a list or a numpy array.")

		self.dfLabel = dfLabel
		#initialize cluster centroid
		random.seed(self.random_state)
		self.initialize_k_cluster_centroids()
		# Update Step
		iteration = 0
		stationary_state = False
		while not stationary_state:
			while iteration < self.max_iterations:
				self.update_cluster_membership()
				if self.update_cluster_centroids() == 1:
					stationary_state = True
					break # terminate if centroids didn't change
				iteration += 1
			if iteration == self.max_iterations:
				print("Termination criteria not met. Consider increasing max_iterations. Reinitialization...")
				self.K = self.constK
				self.initialize_k_cluster_centroids()
				iteration = 0


		self.labels_ = np.argmax(self.clusterMembership, axis=1)










def main():
	# Generate a synthetic 2D dataset with continuous variables
	df = create_binary_class_boundary_spiral(500)
	# Fit K-Means to the data with two clusters
	Mag_Kmeans = MagKmeans(n_clusters = 10, random_state=0)


	# Standardize the features
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(df[['X','Y']].values)


	Mag_Kmeans.fit(X_scaled, df['Label'].values, R = 1)

	# Get cluster centroids and labels
	centroids = scaler.inverse_transform(Mag_Kmeans.cluster_centers_)
	predicted_labels = Mag_Kmeans.labels_
	# Create a scatter plot to visualize the clustering


	# Create a scatter plot for each group
	colors = cm.get_cmap('viridis', Mag_Kmeans.K)
	fig, ax = plt.subplots(1, 1)
	# Define the significance level (alpha) and degrees of freedom (df)
	alpha = 0.2  # For a 95% confidence level
	degofFreedom = 2       # Degrees of freedom (can be adjusted)
	# Calculate the chi-squared critical value
	chi2_critical_value = stats.chi2.ppf(1 - alpha, degofFreedom)

	for c, l, i  in zip(colors(np.unique( predicted_labels )),  np.unique( predicted_labels ), range(Mag_Kmeans.K)):
		condition_plus = (df['Label'] == 1) & (predicted_labels == l)
		condition_minus = (df['Label'] == -1) & (predicted_labels == l)
		ax.scatter(df[ condition_plus]['X'], df[condition_plus]['Y'], marker='o', facecolors='none', label="point with label 1 in cluster" + str(l),  edgecolor = c) #
		ax.scatter(df[condition_minus]['X'], df[condition_minus]['Y'], marker='o', label="point with label -1 in cluster" + str(l), color = c) 

		cluster_points = df[['X','Y']].values[predicted_labels == i]
		covariance_matrix = np.cov(cluster_points, rowvar=False)
		
		# Calculate the center and width/height of the ellipse
		ellipse_center = centroids[i]
		width, height = 2 * np.sqrt(chi2_critical_value * np.linalg.eigvals(covariance_matrix))
		
		# Create and plot the ellipse
		ellipse = plt.matplotlib.patches.Ellipse(ellipse_center, width, height, fill=False, color = c, label = 'Confidence interval for cluster' + str(l))
		fig.gca().add_patch(ellipse)


	#plt.scatter(df[:, 0], df[:, 1], c=predicted_labels, cmap='rainbow')
	ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red', label='Centroids')
	print(Mag_Kmeans.K)
	ax.set_title('K-Means Clustering with Discretized 2D Data')
	#ax.set_xlim(-1,11)
	#ax.set_ylim(-1,11)
	ax.set_xlabel('Feature 1')
	ax.set_ylabel('Feature 2')
	ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
	fig.savefig( 'Plots/Convex.png', bbox_inches='tight')






if __name__ == '__main__':
	main()












