import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import math
import pandas as pd

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

def categorize_values(values, critical_values):
	"""
	Categorize each value in `values` based on the critical thresholds defined in `critical_values`.

	Parameters:
	- values (list of float): The function values to categorize.
	- critical_values (list of float): The critical values defining the thresholds.

	Returns:
	- list of int: The list of labels corresponding to the categories for each value.
	"""
	# Add a very large number to handle the upper boundary easily
	critical_values = sorted(critical_values) + [float('inf')]

	def find_label(value):
		"""Determine the label for a single value based on critical_values."""
		for i, threshold in enumerate(critical_values):
			if value < threshold:
				return i
		return len(critical_values)  # This line is theoretically unreachable

	# Apply the find_label function to each value in the input list
	labels = [find_label(value) for value in values]
	return labels


class eventEstimation(object):
	def __init__(self, qoi_function, domains, output_range, numIntervals):
		self.qoi_function = qoi_function
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

def quantity_interest(input_variable):
	lamb = input_variable[0]
	w = input_variable[1]
	return 1/lamb * math.log(1/(1 - w))



def main():
	event = [[1.1,1.2], [0.1, 0.2]]
	domains = [[1,3],[0,1]]
	sol = eventEstimation(quantity_interest,domains, output_range = [0,5], numIntervals = 20)
	sol.estimation(event, n = 5000)
	event = [[1.1,1.15], [0.1, 0.2]]
	sol.estimation(event,initialization = False)




if __name__ == '__main__':
  main()
