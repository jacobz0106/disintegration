
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
import pandas as pd
from lotkaVolterra import *
import os

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from multiprocessing import cpu_count

def single_generate_POF(i, n, numIntervals, out_range,repeat, domains, quantity_of_interest, gradientFunction, SIP_Data_Multi,outSuffix, method):
	critical_values = np.linspace(out_range[0], out_range[1], numIntervals + 1)[1:-1]
	filenamedf = f'../data/{outSuffix}/df_Train_size{n}_interval_{len(critical_values)+1}_repeat{i}_{method}.csv'
	filenamedQ = f'../data/{outSuffix}/dQ_Train_size{n}_interval_{len(critical_values)+1}_repeat{i}_{method}.csv'
	if os.path.exists(filenamedQ):
		return (i, n, 'exists')

	dataSIP = SIP_Data_Multi(quantity_of_interest, gradientFunction, critical_values, len(domains), *domains)
	if method == 'POF':
		dataSIP.generate_POF(n=n, CONST_a=2, iniPoints=5, sampleCriteria='k-dDarts')
	else:
		dataSIP.generate_Uniform(n)
	df = dataSIP.df
	dQ = dataSIP.Gradient

	df.to_csv(filenamedf, header=True)
	np.savetxt(filenamedQ, np.vstack(dQ).astype(float), delimiter=",")
	return (i, n, 'generated')

def generate_POF_parallel(quantity_of_interest, gradientFunction,N, numIntervals,out_range, domains,outSuffix,repeat=10, max_workers=None, method = 'POF'):


	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = []
		for i in range(repeat):
			for n in N:
				futures.append(executor.submit(
					single_generate_POF, i, n, numIntervals, out_range,repeat, domains,
					quantity_of_interest, gradientFunction, SIP_Data_Multi,outSuffix, method
				))

		for future in tqdm(futures, desc="Generating POF data"):
			i, n, status = future.result()
			print(f"[Repeat {i}, N={n}]: {status}")



##### ------------------------ ################
def main():

	if len(sys.argv) != 4:
		raise ValueError('not enough argument')

	#example, numintervals, sample_method  = ['function2_PPSVMG', 'function2_NN', Brusselator, Elliptic, Function1, Function2], sample method 
	example, numIntervals, sample_method  = sys.argv[1:5]
	numIntervals = int(numIntervals)
	N = [100,120,140,160,180, 200,250,300,400,600,800,1000, 1400,1600,2000]
	repeat = 30
	n = 5000
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
	out_range = [min(np.array(dataSIP.df['f']).reshape(-1)),max(np.array(dataSIP.df['f']).reshape(-1))]
	critical_values = np.linspace(out_range[0], out_range[1], numIntervals + 1)[1:-1]

	# ------------------------------------------------------------------------------#
	print('---')
	generate_POF_parallel(quantity_of_interest, gradientFunction,N, numIntervals,out_range, domains,outSuffix = example,repeat = repeat, max_workers= cpu_count(), method = sample_method)





if __name__ == '__main__':
  main()