from dataGeneration import *
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
import time
from sklearn.neural_network import MLPRegressor
from scipy.stats import truncnorm
from sklearn.neighbors import KernelDensity
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import sys
import math
import pandas as pd
from CBP import *
from lotkaVolterra import *
from multiprocessing import cpu_count, get_context, Lock
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sqlitedict import SqliteDict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial
import warnings
from sklearn.exceptions import ConvergenceWarning

with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category=ConvergenceWarning)


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
	'max_iter':[1000,1500,3000], 
}






def perform_grid_search_cv(model, param_grid, X, y, cv=5,n_jobs=1):
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
	min_class_count = np.min(np.bincount(y))
	cv = min(5, min_class_count)
	if cv <= 2:
		cv = 2
	grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', verbose = 0, n_jobs = n_jobs)
	# Fit the grid search to the data
	grid_search.fit(X, y)
	# Get the best model with tuned hyperparameters
	best_model = grid_search.best_estimator_

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
	x_grid = np.linspace(min(empiricalOutput.reshape(-1)), max(empiricalOutput.reshape(-1)), 1000)

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


def accuracyComparisonNaive(out_suffix,quantity_of_interest, gradientFunction, event, N, domains, critical_values,kde_cdf, repeat  = 20):
	global trunc_a, trunc_b, numIntervals, loc, scale
	mseMatrix = np.zeros( shape = (repeat, len(N)) )
	estimationMatrix = np.zeros( shape = (repeat, len(N)) )

	for i, n in enumerate(N):
		for r in range(repeat):
			dataSIP = SIP_Data_Multi(quantity_of_interest, gradientFunction, critical_values, len(domains) , *domains)
			dataSIP.generate_Uniform(n, Gradient = False)
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
	filenamePredict = f'../Results/BrusselatorSimulation/Estimation_{out_suffix}_interval_{len(critical_values)+1}_Naive.csv'
	header_string = ','.join(str(i) for i in N)
	np.savetxt(filenamePredict, estimationMatrix, delimiter=",", header = header_string)

def event_estimation(function_y,function_Gradient,event, n, domains, critical_values, kde_cdf,repeat = 10):
	estimationMatrix = np.zeros(repeat)

	for r in range(repeat):
		dataSIP = SIP_Data_Multi(function_y,function_Gradient, critical_values, len(domains) , *domains)
		dataSIP.generate_Uniform(n, Gradient = False)
		dfTrain = dataSIP.df.iloc[:, :-2].values

		X_train = dfTrain
		y_train = dataSIP.df['f']


		Labels = categorize_values(y_train, critical_values)
		Within_events = check_points_in_nd_domain(np.array(X_train), np.array(event)[:,0], np.array(event)[:,1])
		event_probability = 0.0
		for equivalenceSpace in np.unique(Labels):
			disintegrationConditional =  np.sum(np.logical_and(Labels == equivalenceSpace, Within_events))/np.sum(Labels == equivalenceSpace)
			equivalenceSpace_probability = equivalenceSpaceProbability(kde_cdf, critical_values, equivalenceSpace)
			event_probability += equivalenceSpace_probability * disintegrationConditional
		estimationMatrix[r] = event_probability
		print(r)
	print(np.mean(estimationMatrix))


#### wrapper for parallel -------------------------------------------------------------------------------------
# lock = Lock()
# shared_lock = None

# def initializer(l):
# 	global shared_lock
# 	shared_lock = l  # available to all worker processes

# # Wrapped single_run to include SqliteDict saving
# def single_run_sqlite(out_suffix, n, r, quantity_of_interest, gradientFunction, model, param_grid, event, domains,
# 					  critical_values, kde_cdf, X_test, y_test, sample_method, grid_search,
# 					  OneVsRestWrapper, MLPClassifier, SIP_Data_Multi, check_points_in_nd_domain,
# 					  equivalenceSpaceProbability, perform_grid_search_cv, GridSearchCV, db_path='results.sqlite'):
# 	global shared_lock
# 	key = f"{sample_method}_{out_suffix}_{n}_repeat_{r}_intervals_{len(critical_values) + 1}"
# 	with shared_lock:
# 		with SqliteDict(db_path, flag = 'r') as db:
# 			if key in db:
# 				print(f"⏩ Skipping existing run: {key}")
# 				return r, None, None  # Signal to skip this run

# 	dataSIP = SIP_Data_Multi(quantity_of_interest, gradientFunction, critical_values, len(domains), *domains)
# 	if sample_method == 'POF':
# 		dataSIP.generate_POF(n=n, CONST_a=2, iniPoints=5, sampleCriteria='k-dDarts')
# 	else:
# 		dataSIP.generate_Uniform(n)

# 	Label = dataSIP.df['Label'].values
# 	dfTrain = dataSIP.df.iloc[:, :-2].values
# 	dQ = dataSIP.Gradient

# 	X_train = dfTrain
# 	y_train = Label

# 	if isinstance(model, OneVsRestWrapper):
# 		fit_para = {'dQ': dQ}
# 		if grid_search:
# 			grid_search_obj = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=0)
# 			grid_search_obj.fit(X_train, y_train, **fit_para)
# 			best_model = grid_search_obj.best_estimator_
# 		else:
# 			best_model = model
# 			best_model.fit(X_train, y_train, dQ)
# 	elif isinstance(model, MLPClassifier):
# 		best_model = perform_grid_search_cv(model, param_grid, X_train, y_train)

# 	predictionAccuracy = np.sum(best_model.predict(X_test) == y_test) / len(y_test)

# 	Labels = best_model.predict(X_test)
# 	Within_events = check_points_in_nd_domain(np.array(X_test), np.array(event)[:, 0], np.array(event)[:, 1])

# 	event_probability = 0
# 	for equivalenceSpace in np.unique(Labels):
# 		disintegrationConditional = np.sum(
# 			np.logical_and(Labels == equivalenceSpace, Within_events)) / np.sum(Labels == equivalenceSpace)
# 		equivalenceSpace_probability = equivalenceSpaceProbability(kde_cdf, critical_values, equivalenceSpace)
# 		event_probability += equivalenceSpace_probability * disintegrationConditional
# 	with shared_lock:
# 		with SqliteDict(db_path, autocommit=True) as db:
# 			db[key] = {'accuracy': predictionAccuracy, 'estimation': event_probability}
# 	return r, predictionAccuracy, event_probability

# def accuracyComparison_parallel_repeat(
# 	quantity_of_interest, gradientFunction, model, param_grid, event,
# 	N, domains, critical_values, kde_cdf, out_suffix,
# 	nTest=2000, repeat=20, sample_method='POF', grid_search=True, max_workers=4,
# 	db_path='Results/dic.sqlite'
# 	):
	

# 	ctx = get_context("spawn")  # safer for clusters
# 	lock = ctx.Lock()

# 	# Generate test set once
# 	testSIP = SIP_Data_Multi(quantity_of_interest, gradientFunction, critical_values, len(domains), *domains)
# 	testSIP.generate_Uniform(nTest)
# 	X_test = testSIP.df.iloc[:, :-2].values
# 	y_test = testSIP.df['Label'].values

# 	with ctx.Pool(processes=max_workers, initializer=initializer, initargs=(lock,)) as pool:

# 		# partial is safer than lambda for pickling
# 		task_func = partial(single_run_sqlite,
# 							out_suffix=out_suffix,
# 							quantity_of_interest=quantity_of_interest,
# 							gradientFunction=gradientFunction,
# 							model=model,
# 							param_grid=param_grid,
# 							event=event,
# 							domains=domains,
# 							critical_values=critical_values,
# 							kde_cdf=kde_cdf,
# 							X_test=X_test,
# 							y_test=y_test,
# 							sample_method=sample_method,
# 							grid_search=grid_search,
# 							OneVsRestWrapper=OneVsRestWrapper,
# 							MLPClassifier=MLPClassifier,
# 							SIP_Data_Multi=SIP_Data_Multi,
# 							check_points_in_nd_domain=check_points_in_nd_domain,
# 							equivalenceSpaceProbability=equivalenceSpaceProbability,
# 							perform_grid_search_cv=perform_grid_search_cv,
# 							GridSearchCV=GridSearchCV,
# 							db_path=db_path)

# 		args = [(n, r) for n in N for r in range(repeat)]
# 		results = list(tqdm(pool.starmap(task_func, args), total=len(args)))



# def accuracyComparison_parallel_repeat(
# 	quantity_of_interest, gradientFunction, model, param_grid, event,
# 	N, domains, critical_values, kde_cdf, out_suffix,
# 	nTest=2000, repeat=20, sample_method='POF', grid_search=True, max_workers=4,
# 	db_path='Results/dic.sqlite'
# 	):
# 	# Generate test set once
# 	testSIP = SIP_Data_Multi(quantity_of_interest, gradientFunction, critical_values, len(domains), *domains)
# 	testSIP.generate_Uniform(nTest)
# 	X_test = testSIP.df.iloc[:, :-2].values
# 	y_test = testSIP.df['Label'].values

# 	ctx = get_context("spawn")  # safer for clusters
# 	lock = ctx.Lock()

# 	with ProcessPoolExecutor(max_workers=max_workers) as executor:
# 		futures = []
# 		for i, n in enumerate(N):
# 			for r in range(repeat):
# 				f = executor.submit(
# 					single_run_sqlite, out_suffix,n, r, quantity_of_interest, gradientFunction, model, param_grid, event,
# 					domains, critical_values, kde_cdf, X_test, y_test, sample_method, grid_search,
# 					OneVsRestWrapper, MLPClassifier, SIP_Data_Multi, check_points_in_nd_domain,
# 					equivalenceSpaceProbability, perform_grid_search_cv, GridSearchCV, db_path
# 				)
# 				futures.append((i, f))

# 		for i, f in tqdm(futures, desc="Processing", total=len(futures)):
# 			r, acc, est = f.result()


def run_single_task(arg):
	return single_run_sqlite(*arg)

def single_run_sqlite(out_suffix, n, r, quantity_of_interest, gradientFunction, model, param_grid, event,
					  domains, critical_values, kde_cdf, X_test, y_test, sample_method, grid_search,
					  OneVsRestWrapper, MLPClassifier, SIP_Data_Multi, check_points_in_nd_domain,
					  equivalenceSpaceProbability, perform_grid_search_cv, GridSearchCV, db_keys):
	key = f"{sample_method}_{out_suffix}_{n}_repeat_{r}_intervals_{len(critical_values) + 1}"

	if key in db_keys:
		print(f"Skipping existing run: {key}")
		return None  # Skip

	# ... [same logic as before, excluding any SQLite writing]


	dataSIP = SIP_Data_Multi(quantity_of_interest, gradientFunction, critical_values, len(domains), *domains)
	if sample_method == 'POF':
		dataSIP.generate_POF(n=n, CONST_a=2, iniPoints=5, sampleCriteria='k-dDarts')
	else:
		if isinstance(model, MLPClassifier):
			dataSIP.generate_Uniform(n, Gradient = False)
		else:
			dataSIP.generate_Uniform(n)

	Label = dataSIP.df['Label'].values
	dfTrain = dataSIP.df.iloc[:, :-2].values
	dQ = dataSIP.Gradient

	X_train = dfTrain
	y_train = Label

	if isinstance(model, OneVsRestWrapper):
		fit_para = {'dQ': dQ}
		if grid_search:
			grid_search_obj = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=0)
			grid_search_obj.fit(X_train, y_train, **fit_para)
			best_model = grid_search_obj.best_estimator_
		else:
			best_model = model
			best_model.fit(X_train, y_train, dQ)
	elif isinstance(model, MLPClassifier):
		best_model = perform_grid_search_cv(model, param_grid, X_train, y_train,n_jobs=1)

	predictionAccuracy = np.sum(best_model.predict(X_test) == y_test) / len(y_test)
	Labels = best_model.predict(X_test)
	Within_events = check_points_in_nd_domain(np.array(X_test), np.array(event)[:, 0], np.array(event)[:, 1])

	event_probability = 0
	for equivalenceSpace in np.unique(Labels):
		disintegrationConditional = np.sum(
			np.logical_and(Labels == equivalenceSpace, Within_events)) / np.sum(Labels == equivalenceSpace)
		equivalenceSpace_probability = float(equivalenceSpaceProbability(kde_cdf, critical_values, equivalenceSpace))
		event_probability += equivalenceSpace_probability * disintegrationConditional

	return key, predictionAccuracy, event_probability

def accuracyComparison_parallel_repeat(
	quantity_of_interest, gradientFunction, model, param_grid, event,
	N, domains, critical_values, kde_cdf, out_suffix,
	nTest=2000, repeat=20, sample_method='POF', grid_search=True, max_workers=4,
	db_path='Results/dic.sqlite'):

	# Step 1: Load existing keys before parallel
	with SqliteDict(db_path, autocommit=True) as db:
		db_keys = set(db.keys())

	# Step 2: Setup test data
	testSIP = SIP_Data_Multi(quantity_of_interest, gradientFunction, critical_values, len(domains), *domains)
	testSIP.generate_Uniform(nTest, Gradient=False)
	X_test = testSIP.df.iloc[:, :-2].values
	y_test = testSIP.df['Label'].values

	# Step 3: Prepare args
	args = [
		(out_suffix, n, r, quantity_of_interest, gradientFunction, model, param_grid, event,
		 domains, critical_values, kde_cdf, X_test, y_test, sample_method, grid_search,
		 OneVsRestWrapper, MLPClassifier, SIP_Data_Multi, check_points_in_nd_domain,
		 equivalenceSpaceProbability, perform_grid_search_cv, GridSearchCV, db_keys)
		for n in N for r in range(repeat)
	]

	# Step 4: Run in parallel
	ctx = get_context("spawn")
	results = []
	results_to_write = {}
	last_write_time = time.time()

	with ctx.Pool(processes=max_workers) as pool:
		for result in tqdm(pool.imap_unordered(run_single_task, args), total=len(args)):
			if result is not None:
				key, acc, est = result
				results.append((key, acc, est))
				results_to_write[key] = {'accuracy': acc, 'estimation': est}

			# Step 5: Periodically flush results to SQLite
			if time.time() - last_write_time >= 600:  # 10 minutes
				with SqliteDict(db_path, autocommit=True) as db:
					db.update(results_to_write)
				results_to_write = {}
				last_write_time = time.time()

	# Step 6: Final flush
	if results_to_write:
		with SqliteDict(db_path, autocommit=True) as db:
			db.update(results_to_write)

	return results
	
	# ctx = get_context("spawn")
	# lock = ctx.Lock()

	# testSIP = SIP_Data_Multi(quantity_of_interest, gradientFunction, critical_values, len(domains), *domains)
	# testSIP.generate_Uniform(nTest, Gradient = False)
	# X_test = testSIP.df.iloc[:, :-2].values
	# y_test = testSIP.df['Label'].values

	# args = [
	# 	(out_suffix, n, r, quantity_of_interest, gradientFunction, model, param_grid, event,
	# 	 domains, critical_values, kde_cdf, X_test, y_test, sample_method, grid_search,
	# 	 OneVsRestWrapper, MLPClassifier, SIP_Data_Multi, check_points_in_nd_domain,
	# 	 equivalenceSpaceProbability, perform_grid_search_cv, GridSearchCV, db_path)
	# 	for n in N for r in range(repeat)
	# ]

	# with ctx.Pool(processes=max_workers, initializer=initializer, initargs=(lock,)) as pool:
	# 	#results = list(tqdm(pool.starmap(single_run_sqlite, args), total=len(args)))
	# 	results = []
	# 	for result in tqdm(pool.imap_unordered(run_single_task, args), total=len(args)):
	# 		results.append(result)
	
	# return results




##### ------------------------ ################
def main():




	if len(sys.argv) != 5:
		raise ValueError('not enough argument')

	#example, model, numintervals, sample_method  = ['function2_PPSVMG', 'function2_NN', Brusselator, Elliptic, Function1, Function2], sample method 
	example, model, numIntervals, sample_method  = sys.argv[1:5]
	numIntervals = int(numIntervals)
	n = 5000
	N = [100,120,140,160,180, 200,250,300,400,600,800,1000, 1400,1600,2000]
	nTest = 5000
	repeat = 20

	out_suffix = f'{example}_{model}'
	db_preffix = f'{example}_{model}_{numIntervals}_{sample_method}'

	if example == 'function2':
		# function2 ---------------
		domains = [[-1,1], [-1,1] ]
		event = [[0,0.8],[-0.7,0.5]] 
		quantity_of_interest=function2
		gradientFunction=Gradient_f2

	elif example == "brusselator":
		domains = [[0.7,1.5],[2.75,3.25],[0,2]]
		event = [[1,1.2],[2.75,3.0],[0.2,1.9]]
		quantity_of_interest=integral_3D
		gradientFunction=DQ_Dlambda_3D
	elif example == "lotka":
		domains = [
		[0.1, 2],
		[0.1, 2],
		[0.1, 2],
		[0.25, 0.75],
		[0.25, 0.75],
		[0.25, 0.75],
		[0.25, 0.75],
		[0.25, 0.75],
		[0.25, 0.75]
		]
		event = [
		[0.2, 1.8],
		[0.4, 1.9],
		[1.2, 1.7],
		[0.3, 0.7],
		[0.3, 0.65],
		[0.35, 0.65],
		[0.25, 0.65],
		[0.35, 0.65],
		[0.3, 0.65]
		]
		lotka = lotkaVolterra()
		quantity_of_interest=lotka.quantity_interest
		gradientFunction=lotka.Gradients
	else:
		raise ValueError('Not implemented.')


	dataSIP = SIP_Data(quantity_of_interest, gradientFunction, 1, len(domains) , *domains)
	dataSIP.generate_Uniform(n, Gradient = False)
	dfTrain = dataSIP.df.iloc[:, :-2].values
	kde_cdf = kde_estimation(np.array(dataSIP.df['f']).reshape(-1,1))
	out_range = [min(np.array(dataSIP.df['f']).reshape(-1)),max(np.array(dataSIP.df['f']).reshape(-1))]
	critical_values = np.linspace(out_range[0], out_range[1], numIntervals + 1)[1:-1]


	if model == 'PPSVMG':	
		base = GMSVM_reduced(clusterSize = 6,ensembleNum=1,C = 0.1,  K = 1, reduced = False, similarity = 0.5)
		model = OneVsRestWrapper(base)
		param_grid= {  
		'clusterSize': [6, 8],    
		'ensembleNum': [1], 
		'C':[0.1,1],     
		'K':[0.1,1,10]
		  }
		grid_search = False
		max_workers =cpu_count()
	else:
		model = MLPClassifier(early_stopping=True, validation_fraction=0.1)
		## - MLPClassifier:
		param_grid = {
		  'hidden_layer_sizes': [(50, 50), (100, 100), (50, 100, 50)],  # Architecture of hidden layers
		  'activation': ['logistic', 'tanh', 'relu'],  # Activation function
		  'alpha': [0.0001, 0.001, 0.01],  # L2 regularization term
		  'learning_rate_init': [0.001, 0.01, 0.1],  # Initial learning rate
		  'max_iter': [3000, 5000, 10000],  # Maximum number of iterations
		}
		grid_search = True
		max_workers = 5


	#event_estimation(quantity_of_interest,gradientFunction,event, n, domains, critical_values, kde_cdf,repeat = 10)

	accuracyComparison_parallel_repeat(
	quantity_of_interest=quantity_of_interest,
	gradientFunction=gradientFunction,
	model=model,
	param_grid=param_grid,
	event=event,
	N=N,
	domains=domains,
	critical_values=critical_values,
	kde_cdf=kde_cdf,
	out_suffix=out_suffix,
	nTest=nTest,
	repeat=repeat,
	sample_method=sample_method,
	grid_search=grid_search,
	max_workers=max_workers,  # set number of parallel processes here,
	db_path=f'../Results/{db_preffix}.sqlite'
	)

	#accuracyComparisonNaive('function2_PPSVMG',function2, Gradient_f2, event, N, domains, critical_values,kde_cdf, repeat  = 20)








	#kde_estimation(domains)
if __name__ == '__main__':
  main()
