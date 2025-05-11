from dataGeneration import *
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split

from sklearn.neural_network import MLPRegressor
from scipy.stats import truncnorm
from sklearn.neighbors import KernelDensity
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import sys
import math
import pandas as pd
from CBP import *

trunc_a = 2
trunc_b = 4.5 
loc = 3.3
scale = 1



# Function to create model, required for KerasRegressor
def sequential_model(layers=1, neurons=10,activation = 'relu'):
    model = Sequential()
    model.add(Dense(neurons, input_dim=3, activation=activation))  # Assuming input features are 10
    for i in range(layers-1):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for regression
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Wrap the model with KerasRegressor

# Define the grid search parameters
param_grid_nn = {
    'hidden_layer_sizes': [(50, 50), (100, 100), (50, 100, 50)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'alpha': [0.0001, 0.001],
    'max_iter':[500,1000,1500], 
}



## - MLPClassifier:
param_grid_MLP = {
  'hidden_layer_sizes': [(50, 50), (100, 100), (50, 100, 50)],  # Architecture of hidden layers
  'activation': ['logistic', 'tanh', 'relu'],  # Activation function
  'alpha': [0.0001, 0.001, 0.01],  # L2 regularization term
  'learning_rate_init': [0.001, 0.01, 0.1],  # Initial learning rate
  'max_iter': [3000, 5000, 10000],  # Maximum number of iterations
}



def perform_grid_search_cv(model, param_grid, X, y, cv=5):
  """
  Perform hyperparameter tuning using GridSearchCV and cross-validation.

  Parameters:
  - model: Estimator object (e.g., a classifier or regressor).
  - param_grid: Dictionary of hyperparameters to search.
  - X: Feature matrix.
  - y: Target vector.
  - cv: Number of cross-validation folds (default is 5).

  Returns:
  - best_model: The best model with tuned hyperparameters.
  """
  # Create a GridSearchCV object
  grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', verbose = 1, n_jobs = -1)
  # Fit the grid search to the data
  grid_search.fit(X, y)
  # Get the best model with tuned hyperparameters
  best_model = grid_search.best_estimator_
  print(best_model.get_params())

  return best_model


def create_color_dict(n, cmap_name='viridis'):
	"""
	Create a color dictionary mapping integers 0 to n to colors from a specified matplotlib colormap.
	
	Parameters:
	- n (int): The maximum key integer.
	- cmap_name (str): The name of the colormap to use.

	Returns:
	- dict: A dictionary with integer keys and color codes as values.
	"""
	# Load the colormap
	cmap = plt.get_cmap(cmap_name)
	
	# Generate an array of points from 0 to 1
	points = np.linspace(0, 1, n+1)
	
	# Map these points to colors using the colormap
	colors = cmap(points)
	
	# Convert RGBA colors to hexadecimal
	hex_colors = [matplotlib.colors.rgb2hex(color) for color in colors]
	
	# Create dictionary mapping integers to colors
	color_dict = {i: hex_colors[i] for i in range(n+1)}
	
	return color_dict


def brusselator2Dplot(n,sep = 10):
	domains = [[0.7,1.5], [2.75,3.25], [1.5,1.5]]
	critical_values = np.linspace(3.0, 4.0, sep)
	dataSIP = SIP_Data_Multi(integral_3D, DQ_Dlambda_3D, critical_values, len(domains) , *domains)


	dataSIP.generate_POF(n = n, CONST_a = 1 ,iniPoints = 10, sampleCriteria = 'k-dDarts')
	print(np.unique(np.array(dataSIP.df['Label'])))

	color_dic = create_color_dict(np.max(np.unique(np.array(dataSIP.df['Label']))), 'rainbow')
	fig = plt.figure()

	# Add a 3D subplot
	ax1 = fig.add_subplot()
	plt.scatter(dataSIP.df['X1'], dataSIP.df['X2'], c = [color_dic[x] for x in dataSIP.df['Label']])
	for c, l, r in zip(dataSIP.df[['X1','X2']].values, dataSIP.df['Label'],dataSIP.POFdarts.radius):
		circle = plt.Circle(c, r, facecolor = color_dic[l], edgecolor = 'black', alpha = 0.5)
		ax1.add_patch(circle)
	plt.show()


def check_points_in_nd_domain(points, lower_bounds, upper_bounds):
	"""
	Check if each point in a list of n-dimensional points is within a specified n-dimensional domain.

	Parameters:
	- points (np.ndarray): An array of points, where each row represents a point and each column a dimension.
	- lower_bounds (np.ndarray): An array representing the lower bounds of the domain for each dimension.
	- upper_bounds (np.ndarray): An array representing the upper bounds of the domain for each dimension.

	Returns:
	- np.ndarray: An array of booleans, each indicating whether the corresponding point is within the domain.
	"""
	# Ensure points, lower_bounds, and upper_bounds are numpy arrays for vectorized operations
	points = np.array(points)
	lower_bounds = np.array(lower_bounds)
	upper_bounds = np.array(upper_bounds)

	# Check if all dimensions of each point are within the respective bounds
	is_within_bounds = np.all((points >= lower_bounds) & (points <= upper_bounds), axis=1)
	
	return is_within_bounds


def kde_estimation(empiricalOutput):
	kde = KernelDensity(kernel='linear', bandwidth=0.2).fit(empiricalOutput)
	# Evaluate KDE on a grid
	x_grid = np.linspace(0, 6, 1000)

	# Compute PDF and CDF
	log_pdf = kde.score_samples(x_grid[:, None])
	pdf = np.exp(log_pdf)
	cdf = cumtrapz(pdf, x_grid, initial=0)  # CDF by numerical integration

	# Create CDF interpolation function
	cdf_function = interp1d(x_grid, cdf, kind='linear', fill_value="extrapolate")

	return cdf_function


def equivalenceSpaceProbability(kde_cdf, critical_values, i):
	if i > len(critical_values):
		raise ValueError('index out of bound.')
	if i == 0:
		return kde_cdf(critical_values[0])
	elif i == len(critical_values):
		return 1 - kde_cdf(critical_values[i-1])
	else:
		return kde_cdf(critical_values[i]) - kde_cdf(critical_values[i-1])


class eventEstimation(object):
	def __init__(self, qoiFunction,gradientFunction, domains, output_range, numIntervals):
		self.qoiFunction = qoiFunction
		self.domains = domains
		self.output_range = output_range
		self.numIntervals =numIntervals
		self.critical_values = np.linspace(output_range[0], output_range[1], numIntervals + 2)[1:-1]
		self.kde_cdf = None

	def estimation(self,event, n = 1000, initialization = True):
		if initialization == True:
			points = np.array([np.random.uniform(low, high, n) for low, high in self.domains]).T
			df = pd.DataFrame(points, columns = [f'X{i+1}' for i in range(len(self.domains) )])
			Z = df.apply(self.qoi_function, axis = 1)

			kde_cdf = kde_estimation(np.array(Z).reshape(-1, 1))

			Labels = categorize_values(Z, self.critical_values)
			self.points = points
			self.Labels = Labels
			self.kde_cdf = kde_cdf
		elif self.kde_cdf is None:
			raise ValueError('generate data by initialization first...')

		Within_events = check_points_in_nd_domain(np.array(self.points), np.array(event)[:,0], np.array(event)[:,1])
		event_probability = 0

		for equivalenceSpace in np.unique(self.Labels):
			disintegrationConditional =  np.sum(np.logical_and(self.Labels == equivalenceSpace, Within_events))/np.sum(self.Labels == equivalenceSpace)
			equivalenceSpace_probability = equivalenceSpaceProbability(self.kde_cdf, self.critical_values, equivalenceSpace)
			event_probability += equivalenceSpace_probability * disintegrationConditional
		print(f"With {n} points, {self.numIntervals} discretizations, estimated prob is {event_probability}")
		return event_probability











def accuracyComparison(quantity_of_interest, gradientFunction, model, param_grid,event, N, domains, critical_values,kde_cdf, out_suffix,nTest = 2000, repeat  = 20, sample_method = 'POF', grid_search =True):
	accuracyMatrix = np.zeros( shape = (repeat, len(N)) )
	estimationMatrix = np.zeros( shape = (repeat, len(N)) )
	testSIP = SIP_Data_Multi(quantity_of_interest, gradientFunction, critical_values, len(domains) , *domains)
	testSIP.generate_Uniform(nTest)
	X_test = testSIP.df.iloc[:, :-2].values
	y_test = testSIP.df['Label'].values

	for i, n in enumerate(N):
		for r in range(repeat):
			dataSIP = SIP_Data_Multi(quantity_of_interest, gradientFunction, critical_values, len(domains) , *domains)
			if sample_method == 'POF':
				dataSIP.generate_POF(n = n, CONST_a = 2 ,iniPoints = 5, sampleCriteria = 'k-dDarts')
			else:
				dataSIP.generate_Uniform(n)

			Label = dataSIP.df['Label'].values
			dfTrain = dataSIP.df.iloc[:, :-2].values
			dQ = dataSIP.Gradient

			X_train = dfTrain
			y_train = Label

			if isinstance(model, OneVsRestWrapper):
				fit_para = {'dQ':dQ}
				if grid_search == True:
					grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose = 1)

					# Fit the grid search to the data
					grid_search.fit(X_train, y_train, **fit_para)
					# Get the best model with tuned hyperparameters
					best_model = grid_search.best_estimator_
				else:
					best_model = model
					best_model.fit(X_train, y_train, dQ)
			elif isinstance(model,MLPClassifier):
				best_model = perform_grid_search_cv(model, param_grid, X_train,y_train)
			trainAccuracy = np.sum(best_model.predict(X_train) == y_train)/len(y_train)
			print(trainAccuracy)
			predictionAccuracy = np.sum(best_model.predict(X_test) == y_test)/len(y_test)
			print(predictionAccuracy)
			accuracyMatrix[r, i] = predictionAccuracy


			Labels = best_model.predict(X_test)
			Within_events = check_points_in_nd_domain(np.array(X_test), np.array(event)[:,0], np.array(event)[:,1])

			event_probability = 0

			for equivalenceSpace in np.unique(Labels):
				disintegrationConditional =  np.sum(np.logical_and(Labels == equivalenceSpace, Within_events))/np.sum(Labels == equivalenceSpace)
				equivalenceSpace_probability = equivalenceSpaceProbability(kde_cdf, critical_values, equivalenceSpace)
				event_probability += equivalenceSpace_probability * disintegrationConditional
			estimationMatrix[r, i] = event_probability
			print('n,r:',[n,r])
	filenameTrain = f'../Results/BrusselatorSimulation/Pred_accuracy_{nTest}_{out_suffix}_{len(critical_values)}_intervals_{len(critical_values)}_{sample_method}_.csv'
	filenamePredict = f'../Results/BrusselatorSimulation/Estimation_{nTest}_{out_suffix}_{len(critical_values)}_intervals_{len(critical_values)}_{sample_method}.csv'
	header_string = ','.join(str(i) for i in N)
	np.savetxt(filenameTrain, accuracyMatrix, delimiter=",", header = header_string)
	np.savetxt(filenamePredict, estimationMatrix, delimiter=",", header = header_string)




def accuracyComparisonNaive(event, N, domains, critical_values,kde_cdf, repeat  = 20):
	global trunc_a, trunc_b, numIntervals, loc, scale
	mseMatrix = np.zeros( shape = (repeat, len(N)) )
	estimationMatrix = np.zeros( shape = (repeat, len(N)) )

	for i, n in enumerate(N):
		for r in range(repeat):
			dataSIP = SIP_Data_Multi(integral_3D, DQ_Dlambda_3D, critical_values, len(domains) , *domains)
			dataSIP.generate_Uniform(n)
			dfTrain = dataSIP.df.iloc[:, :-2].values

			X_train = dfTrain
			y_train = dataSIP.df['f']


			Labels = categorize_values(y_train, critical_values)
			Within_events = check_points_in_nd_domain(np.array(X_train), np.array(event)[:,0], np.array(event)[:,1])
			event_probability = 0

			a, b = (trunc_a - loc) / scale, (trunc_b - loc) / scale
			for equivalenceSpace in np.unique(Labels):
				disintegrationConditional =  np.sum(np.logical_and(Labels == equivalenceSpace, Within_events))/np.sum(Labels == equivalenceSpace)
				equivalenceSpace_probability = equivalenceSpaceProbability(kde_cdf, critical_values, equivalenceSpace)
				event_probability += equivalenceSpace_probability * disintegrationConditional
			estimationMatrix[r, i] = event_probability
			print('n,r:',[n,r])
	filenamePredict = f'../Results/BrusselatorSimulation/Estimation_Brusselator2D_interval_{len(critical_values)}_Naive.csv'
	header_string = ','.join(str(i) for i in N)
	np.savetxt(filenamePredict, estimationMatrix, delimiter=",", header = header_string)

def event_estimation(event, n, domains, critical_values, kde_cdf,repeat = 10):
	global trunc_a, trunc_b, numIntervals, loc, scale
	estimationMatrix = np.zeros(repeat)

	for r in range(repeat):
		dataSIP = SIP_Data_Multi(integral_3D, DQ_Dlambda_3D, critical_values, len(domains) , *domains)
		dataSIP.generate_Uniform(n)
		dfTrain = dataSIP.df.iloc[:, :-2].values

		X_train = dfTrain
		y_train = dataSIP.df['f']


		Labels = categorize_values(y_train, critical_values)
		Within_events = check_points_in_nd_domain(np.array(X_train), np.array(event)[:,0], np.array(event)[:,1])
		event_probability = 0

		for equivalenceSpace in np.unique(Labels):
			disintegrationConditional =  np.sum(np.logical_and(Labels == equivalenceSpace, Within_events))/np.sum(Labels == equivalenceSpace)
			equivalenceSpace_probability = equivalenceSpaceProbability(kde_cdf, critical_values, equivalenceSpace)
			event_probability += equivalenceSpace_probability * disintegrationConditional
		estimationMatrix[r] = event_probability
	print(np.mean(estimationMatrix))

########### - Events - ###################



#######

# choose Pn 
# at many pts in Lambda  Q(lambda)
# make kde/histgram

###


def generate_POF(N,domains,critical_values,repeat = 20):
	for i in range(repeat):
		for j in N:
			print([i,j] )
			dataSIP = SIP_Data_Multi(integral_3D, DQ_Dlambda_3D, critical_values, len(domains) , *domains)
			dataSIP.generate_POF(n = n, CONST_a = 1 ,iniPoints = 5, sampleCriteria = 'k-dDarts')








##### ------------------------ ################
def main():
	numIntervals = 10
	n = 5000
	N = [100,120,140,160,180, 200,250,300,400,600,800,1000, 1400,1600,2000]
	nTest = 5000
	repeat = 20
	# function1
	domains = [[-1,1], [-1,1] ]
	event = [[0,0.8],[-0.7,0.5]]
	dataSIP = SIP_Data(function2, Gradient_f2, 1, len(domains) , *domains)
	dataSIP.generate_Uniform(n)
	dfTrain = dataSIP.df.iloc[:, :-2].values
	critical_values = np.linspace(0, 2, numIntervals + 2)[1:-1]
	kde_cdf = kde_estimation(np.array(dataSIP.df['f']).reshape(-1,1))
	base = GMSVM_reduced(clusterSize = 6,ensembleNum=1,C = 0.1,  K = 1, reduced = False, similarity = 0.5)
	model = OneVsRestWrapper(base)
	param_grid_GMSVM_reduced = {  
	'clusterSize': [6, 8],    
	'ensembleNum': [1], 
	'C':[0.1,1],     
	'K':[0.1,1,10]
	  }
	accuracyComparison(function2, Gradient_f2, model, param_grid_GMSVM_reduced,event, N, domains, critical_values,kde_cdf, out_suffix="function2_PPSVMG",nTest = nTest, repeat  = repeat, sample_method = 'POF', grid_search = False)
	




	# global trunc_a, trunc_b
	# sample_method = sys.argv[1]
	# event = [[0.8,1.3],[2.9,3.2],[0.9,1.8]]
	# domains = [[0.7,1.5],[2.75,3.25],[0,2]]


	# sample_method, ML_method, numIntervals, initial = sys.argv[1:5]
	# numIntervals = int(numIntervals)
	# initial = int(initial)

	# if initial == 0:
	# 	# Convert data to DataFrame
	# 	testSIP = SIP_Data(integral_3D, DQ_Dlambda_3D,3.45,len(domains) , *domains)
	# 	#uniform prior on lambda space
	# 	testSIP.generate_Uniform(10000)
	# 	np.save("../empiricalOutput.npy", np.array(testSIP.df['f']).reshape(-1,1))


	# empiricalOutput = np.load("../empiricalOutput.npy")

	# # use kde to remove the bias, use 10,000, use triangular or guadratic kernel, intervals depends on the variance of the kde
	# # 1. use flow to estimate Pd with much fewer samples, 

	# # 2. with Pd, estimate the event


	# kde_cdf = kde_estimation(empiricalOutput)
	# # Generate num_intervals + 2 points (to include and then remove start and end)
	# critical_values = np.linspace(trunc_a, trunc_b, numIntervals + 2)[1:-1]
	# #N = [30,40,50,60,70,80,90, 100,120,140,160,180, 200,250,300, 400,600,800,1000]
	# N = [100,120,140,160,180, 200,250,300,400,600,800,1000, 1400,1600,2000]
	# if ML_method == 'Naive':
	# 	accuracyComparisonNaive(event, N, domains, critical_values,kde_cdf, repeat = 20)
	# else:
	# 	accuracyComparison(event, N, domains, critical_values,kde_cdf,repeat  = 20,nTest = 5000, sample_method = sample_method)

	#brusselator2Dplot(500,sep = 10)

	# plot of how many points in each contour, 
	# crosssection of lotkavoltarra






	#kde_estimation(domains)
if __name__ == '__main__':
  main()
