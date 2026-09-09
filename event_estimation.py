from dataGeneration import *
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time
from sklearn.neural_network import MLPRegressor
from scipy.stats import truncnorm
from sklearn.neighbors import KernelDensity
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import sys
import os
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
from SIR import *

with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Function to create model
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
	- model: Estimator object.
	- param_grid: Dictionary of hyperparameters to search.
	- X: Feature matrix.
	- y: Target vector.
	- cv: Number of cross-validation folds (default is 5).

	Returns:
	- best_model: The best model with tuned hyperparameters.
	"""
	# Create a GridSearchCV object. bincount() may include 0-count entries for
	# gaps in the label space (e.g. SIR with 20 intervals rarely populates every
	# class); ignore those when finding the smallest populated class.
	counts = np.bincount(y)
	min_class_count = int(counts[counts > 0].min())
	cv = min(5, min_class_count)
	if cv < 2:
		cv = 2
	# StratifiedKFold (sklearn's default for classifiers) requires min class
	# count >= n_splits. Fall back to plain KFold when that isn't achievable.
	if min_class_count < 2:
		cv = KFold(n_splits=2, shuffle=True, random_state=0)
	# balanced_accuracy avoids the failure mode where a majority-class-only
	# classifier scores well under 'accuracy' on skewed KDE-derived classes.
	grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='balanced_accuracy', verbose = 0, n_jobs = n_jobs)
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
	bw = 0.2
	# Guard against NaN/inf values in the KDE source. Some cluster runs
	# generated fresh kde_source CSVs with a few pathological (e.g. lotka
	# ODE blow-up) points whose QoI came back non-finite. Silently drop
	# them rather than crash sklearn's KDE fitter.
	arr = np.asarray(empiricalOutput, dtype=float)
	finite_mask = np.isfinite(arr).all(axis=1) if arr.ndim > 1 else np.isfinite(arr)
	if not finite_mask.all():
		n_bad = int((~finite_mask).sum())
		print(f"[kde_estimation] dropping {n_bad} non-finite sample(s) from KDE source")
		arr = arr[finite_mask]
	empiricalOutput = arr
	kde = KernelDensity(kernel='linear', bandwidth=bw).fit(empiricalOutput)
	# Extend integration grid past the observed support by 1.5 bandwidths so
	# the linear-kernel tent (finite support = [x - bw, x + bw]) is fully
	# captured. Without this, up to 20% of KDE mass leaks past
	# [min, max] and gets falsely credited to the last equivalence class via
	# equivalenceSpaceProbability's "1 - kde_cdf(last threshold)" formula.
	# Then normalize so cdf ends at exactly 1.
	x_min = float(min(empiricalOutput.reshape(-1)))
	x_max = float(max(empiricalOutput.reshape(-1)))
	x_grid = np.linspace(x_min - 1.5 * bw, x_max + 1.5 * bw, 4000)

	log_pdf = kde.score_samples(x_grid[:, None])
	pdf = np.exp(log_pdf)
	cdf = cumtrapz(pdf, x_grid, initial=0)
	cdf = cdf / cdf[-1]
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


def accuracyComparisonNaive(example, quantity_of_interest, gradientFunction, event, N, domains, critical_values,kde_cdf, repeat  = 20):
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

			for equivalenceSpace in np.unique(Labels):
				disintegrationConditional =  np.sum(np.logical_and(Labels == equivalenceSpace, Within_events))/np.sum(Labels == equivalenceSpace)
				equivalenceSpace_probability = equivalenceSpaceProbability(kde_cdf, critical_values, equivalenceSpace)
				event_probability += equivalenceSpace_probability * disintegrationConditional
			estimationMatrix[r, i] = event_probability
			print('n,r:',[n,r])
	filenamePredict = f'../Results/Simulation/{example}/Estimation_interval_{len(critical_values)+1}_Naive.csv'
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



#####



#### wrapper for parallel -------------------------------------------------------------------------------------
shared_lock = None

def initializer(lock_):
	global shared_lock
	shared_lock = lock_


def run_single_task(arg):
	return single_run_sqlite(*arg)

def single_run_sqlite(out_suffix, n, r, quantity_of_interest, gradientFunction, model_name, event,
					  domains, critical_values, kde_cdf, X_test, y_test, sample_method, grid_search,db_keys):
	key = f"{sample_method}_{out_suffix}_{n}_repeat_{r}_intervals_{len(critical_values) + 1}"

	if key in db_keys:
		print(f"Skipping existing run: {key}")
		return None  # Skip

	dataSIP = SIP_Data_Multi(quantity_of_interest, gradientFunction, critical_values, len(domains), *domains)
	# out_suffix = f'{example}_{model}'; key = f'{sample_method}_{out_suffix}_...'
	# so parts[1] is the example name, parts[0] is the sample_method
	parts = key.split("_")
	example_dir = parts[1]
	file_df = f'../data/{example_dir}/df_Train_size{n}_interval_{len(critical_values) + 1}_repeat{r}_{sample_method}.csv'
	file_dQ = f'../data/{example_dir}/dQ_Train_size{n}_interval_{len(critical_values) + 1}_repeat{r}_{sample_method}.csv'

	needs_gradient = (model_name == 'PPSVMG')
	data_cached = os.path.exists(file_df) and (not needs_gradient or os.path.exists(file_dQ))

	dQ = None
	if data_cached:
		df = pd.read_csv(file_df, index_col=0).reset_index(drop=True)
		Label = df['Label'].values
		dfTrain = df.iloc[:, :-2].values
		if needs_gradient:
			dq_raw = pd.read_csv(file_dQ, header=None)
			# Older code saved dQ as a Series-of-lists which serialised to a
			# single object column like "[0.011, -0.011]". Detect that shape
			# and force regeneration — SVM_Penalized cannot normalise strings.
			if dq_raw.select_dtypes(include='object').shape[1] > 0:
				print(f"[data-cache] regenerating {file_dQ}: stringified gradient cells detected")
				data_cached = False
			else:
				dQ = dq_raw.values.tolist()
	if not data_cached:
		if sample_method == 'POF':
			dataSIP.generate_POF(n=n, CONST_a=2, iniPoints=5, sampleCriteria='k-dDarts')
		else:
			if model_name == 'PPSVMG':
				dataSIP.generate_Uniform(n)
			else:
				dataSIP.generate_Uniform(n, Gradient = False)

		Label = dataSIP.df['Label'].values
		dfTrain = dataSIP.df.iloc[:, :-2].values
		dQ = dataSIP.Gradient

		# Persist generated data so future runs can skip generation
		os.makedirs(os.path.dirname(file_df), exist_ok=True)
		dataSIP.df.to_csv(file_df)
		if needs_gradient and dQ is not None and dQ is not False:
			# Coerce Series-of-lists / Series-of-arrays to a rectangular
			# numeric ndarray so each row becomes one column-per-partial.
			# Without this, to_csv writes stringified lists that cannot be
			# reloaded (root cause of the SIR PPSVMG readonly-cache bug).
			rows = dQ.values if hasattr(dQ, 'values') else dQ
			dq_arr = np.vstack([np.asarray(row, dtype=float) for row in rows])
			if dq_arr.size > 0:
				pd.DataFrame(dq_arr).to_csv(file_dQ, header=False, index=False)

	X_train = dfTrain
	y_train = Label
	# initiate model inside work instance - --------- modify this part -------------- -  

	if model_name == 'PPSVMG':	
		model = ClassifierChainWrapper(clusterSize = 6,ensembleNum=1,C = 1,  K = 100, reduced = False, similarity = 0.5)
		param_grid= {  
		'clusterSize': [1,3,6],    
		'ensembleNum': [1], 
		'C':[1],     
		'K':[100]
		  }
		grid_search = True
		fit_para = {'dQ': dQ}
		if grid_search:
			grid_search_obj = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=0)
			grid_search_obj.fit(X_train, y_train, **fit_para)
			best_model = grid_search_obj.best_estimator_
		else:
			best_model = model
			best_model.fit(X_train, y_train, dQ)
	else:
		# early_stopping triggers MLPClassifier's internal stratified train/val
		# split, which needs >=2 members per class inside every CV fold. With
		# SIR + POF at 20 intervals, small folds routinely have 1-member
		# classes and the stratified split raises. Disable early_stopping in
		# that case (falls back to full max_iter training).
		counts_y = np.bincount(y_train)
		min_class_train = int(counts_y[counts_y > 0].min())
		use_early_stopping = min_class_train >= 4
		# StandardScaler in front of the MLP: without it, small-magnitude inputs
		# (e.g. SIR's [0,0.35]x[0,0.6]) leave the network stuck predicting the
		# majority class, which silently zeros out unpredicted equivalence
		# classes in the disintegration sum.
		model = Pipeline([
			('scaler', StandardScaler()),
			('mlp', MLPClassifier(early_stopping=use_early_stopping, validation_fraction=0.1)),
		])
		param_grid = {
		  'mlp__hidden_layer_sizes': [(64, 64), (128, 128)],
		  'mlp__activation': ['relu', 'tanh'],
		  'mlp__alpha': [1e-4, 1e-3],
		  'mlp__learning_rate_init': [1e-3, 1e-2],
		  'mlp__max_iter': [3000],
		}
		best_model = perform_grid_search_cv(model, param_grid, X_train, y_train,n_jobs=1)


	predictionAccuracy = np.sum(best_model.predict(X_test) == y_test) / len(y_test)
	Labels = best_model.predict(X_test)
	Within_events = check_points_in_nd_domain(np.array(X_test), np.array(event)[:, 0], np.array(event)[:, 1])

	# Track cond[k] and count[k] per predicted class so downstream analysis
	# can recompute event_probability with a different KDE without needing
	# to refit the classifier or regenerate the test set.
	event_probability = 0
	cond_per_class = {}
	count_per_class = {}
	for equivalenceSpace in np.unique(Labels):
		mask = Labels == equivalenceSpace
		denom = int(mask.sum())
		numer = int(np.logical_and(mask, Within_events).sum())
		cond = float(numer) / float(denom) if denom > 0 else 0.0
		cond_per_class[int(equivalenceSpace)] = cond
		count_per_class[int(equivalenceSpace)] = denom
		equivalenceSpace_probability = float(equivalenceSpaceProbability(kde_cdf, critical_values, equivalenceSpace))
		event_probability += equivalenceSpace_probability * cond

	return key, predictionAccuracy, event_probability, cond_per_class, count_per_class

def _diagnose_db_writability(db_path):
	"""Diagnose SQLite db path writability before starting expensive compute.

	Prints: cwd, PID, SLURM_JOB_ID, absolute db path, parent-dir path,
	existence, writability, and permission modes. Creates the parent dir if
	missing. Attempts a write probe on the database itself. Raises
	PermissionError with a specific message if any writability check fails.
	"""
	import stat
	abs_path   = os.path.abspath(db_path)
	parent_dir = os.path.dirname(abs_path) or "."
	slurm_job  = os.environ.get('SLURM_JOB_ID', 'n/a')

	print(f"[db-preflight] cwd            = {os.getcwd()}")
	print(f"[db-preflight] pid            = {os.getpid()}")
	print(f"[db-preflight] SLURM_JOB_ID   = {slurm_job}")
	print(f"[db-preflight] db_path        = {abs_path}")
	print(f"[db-preflight] parent_dir     = {parent_dir}")

	if not os.path.isdir(parent_dir):
		try:
			os.makedirs(parent_dir, exist_ok=True)
			print(f"[db-preflight] parent_dir     = created")
		except OSError as e:
			raise PermissionError(
				f"cannot create parent dir {parent_dir}: {e}") from e

	try:
		parent_mode = stat.filemode(os.stat(parent_dir).st_mode)
	except OSError as e:
		raise PermissionError(f"cannot stat parent dir {parent_dir}: {e}") from e
	parent_writable = os.access(parent_dir, os.W_OK)
	print(f"[db-preflight] parent mode    = {parent_mode}  writable={parent_writable}")

	# Warn about leftover journal/wal files from a prior crashed run.
	# Their presence means SQLite will roll back everything since the last
	# commit when the next process opens the DB — completed-but-uncommitted
	# work will disappear and re-run. Not deleted automatically because
	# another process could legitimately be mid-transaction.
	for suffix in ('-journal', '-wal', '-shm'):
		sidecar = abs_path + suffix
		if os.path.exists(sidecar):
			try:
				import time as _time
				mtime = _time.strftime('%Y-%m-%d %H:%M:%S',
				                       _time.localtime(os.path.getmtime(sidecar)))
			except OSError:
				mtime = 'unknown'
			print(f"[db-preflight] STALE {suffix:<8} = {sidecar} (mtime {mtime}) — "
			      f"SQLite may roll back uncommitted work from a prior crash")

	db_exists = os.path.exists(abs_path)
	print(f"[db-preflight] db_exists      = {db_exists}")
	if db_exists:
		db_mode     = stat.filemode(os.stat(abs_path).st_mode)
		db_writable = os.access(abs_path, os.W_OK)
		print(f"[db-preflight] db mode        = {db_mode}  writable={db_writable}")
		if not db_writable:
			raise PermissionError(
				f"database file is not writable: {abs_path} (mode {db_mode})")
	if not parent_writable:
		raise PermissionError(
			f"parent directory is not writable: {parent_dir} (mode {parent_mode})")

	# Probe an actual SQLite write/commit so we catch RO mounts or locking
	# problems that os.access cannot see (e.g. Lustre/NFS quirks).
	probe_key = "__preflight_probe__"
	try:
		with SqliteDict(abs_path, autocommit=False) as db:
			db[probe_key] = 1
			db.commit()
			del db[probe_key]
			db.commit()
	except Exception as e:
		raise PermissionError(
			f"SQLite write probe failed on {abs_path}: {e}") from e
	print(f"[db-preflight] write probe    = ok")


def accuracyComparison_parallel_repeat(
	quantity_of_interest, gradientFunction, model_name, event,
	N, domains, critical_values, kde_cdf, out_suffix,
	nTest=2000, repeat=20, sample_method='POF', grid_search=True,
	db_path='Results/dic.sqlite', commit_every=1):

	# Step 0: Fail fast if the results DB cannot be written to.
	_diagnose_db_writability(db_path)

	# Step 1: Load committed keys from the .sqlite file. Opening writable
	# forces SQLite to complete any pending journal rollback first, so
	# db_keys reflects the *committed* state after recovery.
	with SqliteDict(db_path, autocommit=False) as db:
		db_keys = set(db.keys())
	print(f"[db-preflight] committed_keys = {len(db_keys)}  (source: {os.path.abspath(db_path)})")

	# Step 2: Setup test data
	testSIP = SIP_Data_Multi(quantity_of_interest, gradientFunction, critical_values, len(domains), *domains)
	testSIP.generate_Uniform(nTest, Gradient=False)
	X_test = testSIP.df.iloc[:, :-2].values
	y_test = testSIP.df['Label'].values

	# Step 3: Filter out tasks whose keys are already in the .sqlite file so
	# workers never even start those runs. This is the parent-side skip; the
	# `key in db_keys` check inside single_run_sqlite remains as a defence in
	# depth in case db_keys changes between planning and dispatch.
	def _mk_key(n, r):
		return f"{sample_method}_{out_suffix}_{n}_repeat_{r}_intervals_{len(critical_values) + 1}"

	all_tasks = [(n, r) for n in N for r in range(repeat)]
	skipped   = [(n, r) for (n, r) in all_tasks if _mk_key(n, r) in db_keys]
	to_run    = [(n, r) for (n, r) in all_tasks if _mk_key(n, r) not in db_keys]
	print(f"[db-preflight] tasks_total    = {len(all_tasks)}")
	print(f"[db-preflight] tasks_skipped  = {len(skipped)}  (already in DB)")
	print(f"[db-preflight] tasks_to_run   = {len(to_run)}")
	if 0 < len(to_run) <= 20:
		for n, r in to_run:
			print(f"[db-preflight] to_run key     = {_mk_key(n, r)}")

	args = [
		(out_suffix, n, r, quantity_of_interest, gradientFunction, model_name, event,
		 domains, critical_values, kde_cdf, X_test, y_test, sample_method, grid_search, db_keys)
		for (n, r) in to_run
	]

	if not args:
		print("[db-preflight] nothing to do — all tasks already in DB")
		return []

	# Step 4: Run in parallel. Workers only compute + return; only the parent
	# process opens or writes to the SQLite database.
	ctx = get_context("spawn")
	results = []
	pending = []  # results assigned to db but not yet committed
	unsaved = []  # results that could not be committed at all

	# PPSVMG already runs GridSearchCV internally; keep it at 1 to avoid
	# memory blow-up. Other models are lightweight enough to parallelise.
	if model_name == 'PPSVMG':
		max_workers = 1
	else:
		max_workers = max(1, cpu_count() - 1)

	def _flush(db):
		"""Commit pending writes; on failure, move them to unsaved."""
		if not pending:
			return
		try:
			db.commit()
			pending.clear()
		except Exception as e:
			print(f"[db-commit-error] commit failed on {len(pending)} pending items: {e}")
			print(f"[db-commit-error] unsaved keys: {pending}")
			unsaved.extend(pending)
			pending.clear()
			raise

	with ctx.Pool(processes=max_workers) as pool:
		db = SqliteDict(db_path, autocommit=False)
		try:
			for result in tqdm(pool.imap_unordered(run_single_task, args), total=len(args)):
				if result is None:
					continue
				key, acc, est, cond_per_class, count_per_class = result
				results.append((key, acc, est))
				try:
					db[key] = {
						'accuracy': acc,
						'estimation': est,
						'cond_per_class': cond_per_class,
						'count_per_class': count_per_class,
					}
					pending.append(key)
				except Exception as e:
					print(f"[db-write-error] failed to stage key {key}: {e}")
					unsaved.append(key)
					continue
				if len(pending) >= commit_every:
					_flush(db)
			# Final flush of any remaining pending writes.
			_flush(db)
		finally:
			# Best-effort commit even if the main loop raised — otherwise up to
			# commit_every recent results would be rolled back by SQLite on the
			# next open (via the -journal file), causing completed work to
			# re-run. This is the safeguard that makes db_keys reliable.
			if pending:
				try:
					db.commit()
					pending.clear()
				except Exception as e:
					print(f"[db-commit-error] finally-safeguard commit failed: {e}")
					print(f"[db-commit-error] unsaved keys: {pending}")
					unsaved.extend(pending)
					pending.clear()
			try:
				db.close()
			except Exception as e:
				print(f"[db-close-error] {e}")
			if unsaved:
				print(f"[db-summary] {len(unsaved)} result(s) NOT persisted: {unsaved}")
			else:
				print(f"[db-summary] all {len(results)} result(s) committed")

	return results
	




def PPSVMG_test(n,nTest, event, quantity_of_interest, gradientFunction, critical_values, domains, sample_method = 'POF'):
	print('generate test...')
	testSIP = SIP_Data_Multi(quantity_of_interest, gradientFunction, critical_values, len(domains), *domains)
	testSIP.generate_Uniform(nTest, Gradient=False)
	X_test = testSIP.df.iloc[:, :-2].values
	y_test = testSIP.df['Label'].values
	dataSIP = SIP_Data_Multi(quantity_of_interest, gradientFunction, critical_values, len(domains), *domains)
	print('generate train...')
	if sample_method == 'POF':
		dataSIP.generate_POF(n=n, CONST_a=2, iniPoints=5, sampleCriteria='k-dDarts')
	else:
		print('uniform')
		dataSIP.generate_Uniform(n)
	print('end')
	Label = dataSIP.df['Label'].values
	dfTrain = dataSIP.df.iloc[:, :-2].values
	dQ = dataSIP.Gradient

	X_train = dfTrain
	y_train = Label

	fit_para = {'dQ': dQ}
	base = GMSVM_reduced(clusterSize = 3,ensembleNum=1,C = 1,  K = 100, reduced = False, similarity = 0.5)
	model = OneVsRestWrapper(base)
	best_model = model
	best_model.fit(X_train, y_train, dQ)
	predictionAccuracy = np.sum(best_model.predict(X_test) == y_test) / len(y_test)
	y_pred = best_model.predict(X_test)
	# Scatter plot for predicted labels

	plt.figure(figsize=(12, 5))

	plt.subplot(1, 2, 1)
	plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=40)
	plt.title("Predicted Labels")
	plt.xlabel("x1")
	plt.ylabel("x2")

	# Scatter plot for true labels
	plt.subplot(1, 2, 2)
	plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolor='k', s=40)
	plt.title("True Labels")
	plt.xlabel("x1")
	plt.ylabel("x2")

	plt.tight_layout()
	plt.show()

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
	repeat = 30

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
		gradientFunction=lotka.gradients
	elif example == 'SIR':
		# Λ₁ = [0, 0.35] × [0, 0.6]; β ~ Beta(12,30), γ ~ Beta(6,30); T0=10, T=30
		domains = [[0, 0.35], [0, 0.6]]
		sir = SIR_model(T=30, T0=10)
		quantity_of_interest = sir.quantity_interest
		gradientFunction = sir.gradients
		# Event options (joint probabilities under the Beta priors):
		#   A ~18%: [[0.25, 0.35], [0.06, 0.14]]  — high β, low γ
		#   B ~21%: [[0.20, 0.30], [0.05, 0.15]]  — moderate β, low γ
		#   C ~30%: [[0.22, 0.32], [0.12, 0.22]]  — moderate β and γ
		event = [[0.25, 0.35], [0.06, 0.14]]
	else:
		raise ValueError('Not implemented.')

	if example != 'SIR':
		kde_source_file = f'../data/{example}/kde_source_n{n}.csv'
		if os.path.exists(kde_source_file):
			f_values = pd.read_csv(kde_source_file, index_col=0)['f'].values
		else:
			dataSIP = SIP_Data(quantity_of_interest, gradientFunction, 1, len(domains), *domains)
			dataSIP.generate_Uniform(n, Gradient=False)
			f_values = np.array(dataSIP.df['f'])
			os.makedirs(os.path.dirname(kde_source_file), exist_ok=True)
			dataSIP.df[['f']].to_csv(kde_source_file)
		f_values = f_values[np.isfinite(f_values)]
		kde_cdf = kde_estimation(f_values.reshape(-1, 1))
		out_range = [float(f_values.min()), float(f_values.max())]
		critical_values = np.linspace(out_range[0], out_range[1], numIntervals + 1)[1:-1]
	else:
		# SIR: combine real Surge 2 empirical Q_I with simulated Q_I under the Beta
		# priors on (beta, gamma). The empirical support alone [~4e-4, ~3e-3] is
		# much narrower than the simulated support (up to ~0.02), which collapses
		# most simulated samples into the last equivalence class. Widening the
		# KDE support keeps the classifier training set balanced.
		emp_path = '../data/SIR/empericalData/Surge2_QoI_30_days.csv'
		f_emp = pd.read_csv(emp_path)['case_prop_diff_over_T'].values

		sim_cache = f'../data/SIR/qoi_sim_n{n}.csv'
		if os.path.exists(sim_cache):
			f_sim = pd.read_csv(sim_cache, index_col=0)['f'].values
		else:
			sir_prior = SIR_model(T=30, T0=10)
			rng = np.random.default_rng(42)
			betas  = rng.beta(12, 30, n).clip(0, 0.35)
			gammas = rng.beta(6,  30, n).clip(0, 0.6)
			f_sim = np.array([sir_prior.quantity_interest([b, g])
			                  for b, g in zip(betas, gammas)])
			os.makedirs(os.path.dirname(sim_cache), exist_ok=True)
			pd.DataFrame({'f': f_sim}).to_csv(sim_cache)

		f_values = np.concatenate([f_emp, f_sim])
		# Drop non-finite Q values before fitting KDE (SIR ODE may blow up
		# for a very high-beta / low-gamma sample).
		f_values = f_values[np.isfinite(f_values)]
		bw = 1.06 * f_values.std() * len(f_values) ** (-0.2)
		kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(f_values.reshape(-1, 1))
		# Extend the integration grid past the observed support by 6 bandwidths
		# so the Gaussian tails are captured; without this ~15% of KDE mass
		# leaks past [f_values.min(), f_values.max()] and, via the
		# "1 - kde_cdf(last threshold)" formula in equivalenceSpaceProbability,
		# gets falsely dumped into the last equivalence class. Also normalize
		# so cdf ends at exactly 1 (the truncated integral rounds off).
		xL = float(f_values.min()) - 6.0 * bw
		xR = float(f_values.max()) + 6.0 * bw
		x_grid = np.linspace(xL, xR, 4000)
		log_pdf = kde.score_samples(x_grid[:, None])
		pdf = np.exp(log_pdf)
		cdf_vals = cumtrapz(pdf, x_grid, initial=0)
		cdf_vals = cdf_vals / cdf_vals[-1]
		kde_cdf = interp1d(x_grid, cdf_vals, kind='linear', fill_value='extrapolate')
		out_range = [float(f_values.min()), float(f_values.max())]
		critical_values = np.linspace(out_range[0], out_range[1], numIntervals + 1)[1:-1]

	#event_estimation(quantity_of_interest,gradientFunction,event, n, domains, critical_values, kde_cdf,repeat = 10)

	# db_path may be overridden by env var EVENT_DB_PATH so batch scripts can
	# redirect results without editing the source. Do NOT default to a scratch
	# path here — checkpointing between scratch and shared FS is not yet handled.
	db_path = os.environ.get('EVENT_DB_PATH', f'../Results/{db_preffix}.sqlite')

	accuracyComparison_parallel_repeat(
	quantity_of_interest=quantity_of_interest,
	gradientFunction=gradientFunction,
	model_name=model,
	event=event,
	N=N,
	domains=domains,
	critical_values=critical_values,
	kde_cdf=kde_cdf,
	out_suffix=out_suffix,
	nTest=nTest,
	repeat=repeat,
	sample_method=sample_method,
	grid_search=True,
	db_path=db_path,
	)

	# accuracyComparisonNaive(example, quantity_of_interest, gradientFunction, event, N, domains, critical_values,kde_cdf, repeat  = 30)

	# print('---')
	# PPSVMG_test(1000,2000, event, quantity_of_interest, gradientFunction, critical_values, domains, sample_method= 'Random')






if __name__ == '__main__':
  main()
