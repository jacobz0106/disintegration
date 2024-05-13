from dataGeneration import *
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split

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


def brusselator2Dplot(n):
	domains = [[0.7,1.5], [2.75,3.25], [1.5,1.5]]
	critical_values = np.linspace(3.0, 4.0, 10)
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



def accuracyComparison(event, N, domains, critical_values,nTest = 2000, repeat  = 20, sample_method = 'POF'):
	accuracyMatrix = np.zeros( shape = (repeat, len(N)) )
	estimationMatrix = np.zeros( shape = (repeat, len(N)) )
	for i, n in enumerate(N):
		for r in range(repeat):
			mlp_classifier = MLPClassifier()
			dataSIP = SIP_Data_Multi(integral_3D, DQ_Dlambda_3D, critical_values, len(domains) , *domains)
			if sample_method == 'POF':
				dataSIP.generate_POF(n = n, CONST_a = 1 ,iniPoints = 10, sampleCriteria = 'k-dDarts')
			else:
				dataSIP.generate_Uniform(n)

			# MLP predict event probability 
			Label = dataSIP.df['Label'].values
			dfTrain = dataSIP.df.iloc[:, :-2].values
			dQ = dataSIP.Gradient

			X_train = dfTrain
			y_train = Label

			dataSIP.generate_Uniform(nTest)
			X_test = dataSIP.df.iloc[:, :-2].values
			y_test = dataSIP.df['Label'].values

			best_model = perform_grid_search_cv(mlp_classifier, param_grid_MLP, X_train,y_train)
			trainAccuracy = np.sum(best_model.predict(X_train) == y_train)/len(y_train)
			predictionAccuracy = np.sum(best_model.predict(X_test) == y_test)/len(y_test)
			accuracyMatrix[r, i] = predictionAccuracy


			# Event estimation/ Change points to previous randomly sampled

			Labels = best_model.predict(X_test)
			Within_events = check_points_in_nd_domain(np.array(X_test), np.array(event)[:,0], np.array(event)[:,1])

			event_probability = 0
			equivalenceSpace_probability = 1/(len(critical_values)+1)
			for equivalenceSpace in np.unique(Labels):
				disintegrationConditional =  np.sum(np.logical_and(Labels == equivalenceSpace, Within_events))/np.sum(Labels == equivalenceSpace)
				event_probability += equivalenceSpace_probability * disintegrationConditional
			estimationMatrix[r, i] = event_probability
			print('n,r:',[n,r])
	filenameTrain = f'../Results/BrusselatorSimulation/Train_accuracy_{nTest}_Brusselator2D_interval_{len(critical_values)+1}_{sample_method}.csv'
	filenamePredict = f'../Results/BrusselatorSimulation/Estimation_{nTest}_Brusselator2D_interval_{len(critical_values)+1}_{sample_method}.csv'
	header_string = ','.join(str(i) for i in N)
	np.savetxt(filenameTrain, accuracyMatrix, delimiter=",", header = header_string)
	np.savetxt(filenamePredict, estimationMatrix, delimiter=",", header = header_string)




def main():
	event = [[1.0,1.2],[2.8,3.1],[1.0,2.0]]
	domains = [[0.7,1.5], [2.75,3.25], [1.5,1.5]]
	critical_values = np.linspace(3.0, 4.0, 5)
	N = [100,200, 300, 400,600,800,1000]
	accuracyComparison(event, N, domains, critical_values,repeat  = 20,sample_method = 'POF')
	accuracyComparison(event, N, domains, critical_values,repeat  = 20,sample_method = 'Random')




if __name__ == '__main__':
  main()
