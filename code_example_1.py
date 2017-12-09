# Author: Moises Martinez
# Conference: Codemotion 2017
# Title: Utilizar Machine Learning y no morir en el intento
# Email: momartinm@gmail.com

import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def getPropertyType( pt ):	
        if (pt == 'D'):
		return 1 #Detached
   	elif (pt == 'S'):
		return 2 #Semi-Detached
	elif (pt == 'T'):
		return 3 #Terraced
	elif (pt == 'F'):
		return 4 #Flats/Maisonettes
	else:
		return 5 #Other

def getAge( a ):
	if (a == 'Y'):
		return 1 #Y = a newly built property
	else:
		return 0 #N = an established residential building

def getDuration( d ):
	if (d == 'F'):
		return 1 #Freehold
	else:
		return 0 #Leasehold

def isInLondon( name ):
	if ('LONDON' in name.upper()):
		return 1
	else:
		return 0

def showResults( data_y_pred, data_y_test ):
        print("Mean squared error: %.2f" % mean_squared_error(data_y_test, data_y_pred))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(data_y_test, data_y_pred))

        # Plot outputs
      	plt.plot(data_y_pred, data_y_test, 'ro')
	plt.plot([0,50],[0,50], 'g-')
	plt.xlabel('pred')
	plt.ylabel('test')
	plt.savefig('output.png')

def generateExample( data, cities, mode ):
	if (mode == 1):			
		return [data[2][5:7], getPropertyType(data[4][1:-1]), getDuration(data[6][1:-1]), getAge(data[7][1:-1]), isInLondon(data[12][1:-1]), cities[data[12][1:-1]]]
	elif (mode == 2):
	        return [getAge(data[7][1:-1])]
        elif (mode == 3):
                return [getPropertyType(data[4][1:-1])]
	        
	return [getPropertyType(data[4][1:-1]), getDuration(data[6][1:-1]), isInLondon(data[12][1:-1])]

def incrementalLearning( path, buffer_size, model, mode ):

	cities = {}

	data_X_train = []
	data_y_train = []

	data_X_test = []
	data_y_test = []

	start_year = 1995
	previous_year = start_year

	with open(path) as infile: 
		while True:
			lines = infile.readlines(buffer_size)
			if not lines:
				break
			for line in lines:
				data = line.split(',')
				# Getting current year for the example
				year = int(data[2][0:4]) 
				
				# Generating code for a new city
				if not cities.has_key(data[12][1:-1]):
					cities[data[12][1:-1]] = len(cities)+1				
                                
                                # Generating training set
				if (previous_year < 2015):				
					if (previous_year != year):
					        # Fit linear model with Stochastic Gradient Descent for year
						model.partial_fit(np.array(data_X_train).astype(np.float), np.array(data_y_train).astype(np.float))
						print("Partial fit year %s" % previous_year)
						previous_year = year
                                                # Clean data to collect new year
						data_X_train = []
						data_y_train = []
					data_X_train.append(generateExample(data, cities, mode))
					data_y_train.append(data[1])
			        # Generating training data
				elif (previous_year == 2015):
					data_X_test.append(generateExample(data, cities, mode))
					data_y_test.append(data[1])						
        
	data_y_prediction = model.predict(np.array(data_X_test).astype(np.float))
	print('Coefficients: ', model.coef_)
	showResults(data_y_prediction, np.array(data_y_test).astype(np.float))		

def main( argv ):

        parser = argparse.ArgumentParser(description='Trying to predict the prize of a property in UK')
        parser.add_argument('-f', type=str, nargs=1, help='path to the examples (train/test) file', required=True)
        parser.add_argument('-m', type=int, nargs=1, help='mode of examples are built (0: all data; 1: property age; 2: property type)', default=0, required=False)

        args = parser.parse_args(argv[1:])
        incrementalLearning(args.f[0], 65536, SGDRegressor(loss='squared_loss', penalty='l1', max_iter=25), args.m)

if __name__ == "__main__":
	main(sys.argv)

