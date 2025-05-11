
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





def generate_POF(N,numIntervals,repeat = 10):
	domains = [[0.7,1.5],[2.75,3.25],[0,2]]
	critical_values = np.linspace(2, 4.5, numIntervals + 1)[1:-1]
	for i in range(repeat):
		for n in N:
			print([i,n] )
			filenamedf = f'data/Brusselator/df_Train_size{n}_Brusselator_interval_{len(critical_values)}_repeat{repeat+i}.csv'
			filenamedQ = f'data/Brusselator/dQ_Train_size{n}_Brusselator_interval_{len(critical_values)}_repeat{repeat+i}.csv'
			if os.path.exists(filenamedQ):
				print('exists')

				continue
			else:
				print('generating')
				dataSIP = SIP_Data_Multi(integral_3D, DQ_Dlambda_3D, critical_values, len(domains) , *domains)
				dataSIP.generate_POF(n = n, CONST_a = 1 ,iniPoints = 5, sampleCriteria = 'k-dDarts')
				df = dataSIP.df
				dQ = dataSIP.Gradient
				df.to_csv(filenamedf, header = True)
				np.savetxt(filenamedQ, dQ, delimiter=",")
			



def generate_POF_lotka(N,numIntervals,repeat = 10):
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
	critical_values = np.linspace(0, 2, numIntervals + 1)[1:-1]
	for i in range(repeat):
		for n in N:
			print([i,n] )
			# Specify the file path
			filenamedf = f'data/lotka/df_Train_size{n}_interval_{len(critical_values)+1}_repeat{repeat+i}_POF.csv'
			filenamedQ = f'data/lotka/dQ_Train_size{n}_interval_{len(critical_values)+1}_repeat{repeat+i}_POF.csv'
			# Check if the file exists
			if os.path.exists(filenamedQ):
				print('exists')
				continue
			else:
				print('generating')
				lotka = lotkaVolterra()
				dataSIP = SIP_Data_Multi(lotka.quantity_interest, lotka.Gradients, critical_values, len(domains) , *domains)
				dataSIP.generate_POF(n = n, CONST_a = 2 ,iniPoints = 5, sampleCriteria = 'k-dDarts')
				df = dataSIP.df
				dQ = dataSIP.Gradient
				
				df.to_csv(filenamedf, header = True)
				np.savetxt(filenamedQ, dQ, delimiter=",")





##### ------------------------ ################
def main():

	numIntervals = 10

	N = [100,120,140,160,180, 200,250,300,400,600,800,1000, 1400,1600,2000]

	# ------------------------------------------------------------------------------#

	generate_POF(N,numIntervals,repeat = 20)
	#generate_POF_lotka(N,numIntervals,repeat = 20)


	#kde_estimation(domains)





if __name__ == '__main__':
  main()