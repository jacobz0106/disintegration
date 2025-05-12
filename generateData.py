
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

def single_generate_POF(i, n, numIntervals, out_range,repeat, domains, function_y, function_gradient, SIP_Data_Multi,outSuffix):
	critical_values = np.linspace(out_range[0], out_range[1], numIntervals + 1)[1:-1]
	filenamedf = f'data/{outSuffix}/df_Train_size{n}_interval_{len(critical_values)+1}_repeat{i}.csv'
	filenamedQ = f'data/{outSuffix}/dQ_Train_size{n}_interval_{len(critical_values)+1}_repeat{i}.csv'
	if os.path.exists(filenamedQ):
		return (i, n, 'exists')

	dataSIP = SIP_Data_Multi(function_y, function_gradient, critical_values, len(domains), *domains)
	dataSIP.generate_POF(n=n, CONST_a=2, iniPoints=5, sampleCriteria='k-dDarts')
	df = dataSIP.df
	dQ = dataSIP.Gradient

	df.to_csv(filenamedf, header=True)
	np.savetxt(filenamedQ, dQ, delimiter=",")
	return (i, n, 'generated')

def generate_POF_parallel(function_y, function_gradient,N, numIntervals,out_range, domains,outSuffix,repeat=10, max_workers=None):


	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = []
		for i in range(repeat):
			for n in N:
				futures.append(executor.submit(
					single_generate_POF, i, n, numIntervals, out_range,repeat, domains,
					function_y, function_gradient, SIP_Data_Multi,outSuffix
				))

		for future in tqdm(futures, desc="Generating POF data"):
			i, n, status = future.result()
			print(f"[Repeat {i}, N={n}]: {status}")





# def generate_POF(N,numIntervals,repeat = 10):
# 	domains = [[0.7,1.5],[2.75,3.25],[0,2]]
# 	critical_values = np.linspace(2, 4.5, numIntervals + 1)[1:-1]
# 	for i in range(repeat):
# 		for n in N:
# 			print([i,n] )
# 			filenamedf = f'data/Brusselator/df_Train_size{n}_Brusselator_interval_{len(critical_values)}_repeat{repeat+i}.csv'
# 			filenamedQ = f'data/Brusselator/dQ_Train_size{n}_Brusselator_interval_{len(critical_values)}_repeat{repeat+i}.csv'
# 			if os.path.exists(filenamedQ):
# 				print('exists')

# 				continue
# 			else:
# 				print('generating')
# 				dataSIP = SIP_Data_Multi(integral_3D, DQ_Dlambda_3D, critical_values, len(domains) , *domains)
# 				dataSIP.generate_POF(n = n, CONST_a = 1 ,iniPoints = 5, sampleCriteria = 'k-dDarts')
# 				df = dataSIP.df
# 				dQ = dataSIP.Gradient
# 				df.to_csv(filenamedf, header = True)
# 				np.savetxt(filenamedQ, dQ, delimiter=",")
			



# def generate_POF_lotka(N,numIntervals,repeat = 10):
# 	domains = [
# 		[0.1, 2],
# 		[0.1, 2],
# 		[0.1, 2],
# 		[0.25, 0.75],
# 		[0.25, 0.75],
# 		[0.25, 0.75],
# 		[0.25, 0.75],
# 		[0.25, 0.75],
# 		[0.25, 0.75]
# 	]
# 	critical_values = np.linspace(0, 2, numIntervals + 1)[1:-1]
# 	for i in range(repeat):
# 		for n in N:
# 			print([i,n] )
# 			# Specify the file path
# 			filenamedf = f'data/lotka/df_Train_size{n}_interval_{len(critical_values)+1}_repeat{repeat+i}_POF.csv'
# 			filenamedQ = f'data/lotka/dQ_Train_size{n}_interval_{len(critical_values)+1}_repeat{repeat+i}_POF.csv'
# 			# Check if the file exists
# 			if os.path.exists(filenamedQ):
# 				print('exists')
# 				continue
# 			else:
# 				print('generating')
# 				lotka = lotkaVolterra()
# 				dataSIP = SIP_Data_Multi(lotka.quantity_interest, lotka.Gradients, critical_values, len(domains) , *domains)
# 				dataSIP.generate_POF(n = n, CONST_a = 2 ,iniPoints = 5, sampleCriteria = 'k-dDarts')
# 				df = dataSIP.df
# 				dQ = dataSIP.Gradient
				
# 				df.to_csv(filenamedf, header = True)
# 				np.savetxt(filenamedQ, dQ, delimiter=",")





##### ------------------------ ################
def main():

	numIntervals = 5

	N = [100,120,140,160,180, 200,250,300,400,600,800,1000, 1400,1600,2000]

	# ------------------------------------------------------------------------------#
	out_range = [2, 4.5]
	domains = [[0.7,1.5],[2.75,3.25],[0,2]]
	print(cpu_count())
	generate_POF_parallel(integral_3D, DQ_Dlambda_3D,N, numIntervals,out_range, domains,outSuffix = "Brusselator",repeat=20, max_workers= cpu_count() // 4)

	# domains = [
	# 	[0.1, 2],
	# 	[0.1, 2],
	# 	[0.1, 2],
	# 	[0.25, 0.75],
	# 	[0.25, 0.75],
	# 	[0.25, 0.75],
	# 	[0.25, 0.75],
	# 	[0.25, 0.75],
	# 	[0.25, 0.75]
	# ]
	# out_range = [0,2]
	# generate_POF_parallel(integral_3D, DQ_Dlambda_3D,N, numIntervals,out_range, domains,outSuffix = "lotka",repeat=20, max_workers= max_workers=cpu_count() // 2)


	#kde_estimation(domains)





if __name__ == '__main__':
  main()