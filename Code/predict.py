import cPickle as pickle
import pandas as pd
import numpy as np

def find_comps_predict_one(point, kdtree, data, model, num_comps=100):
	'''
	INPUT: point: data point to be predicted (numpy array)
				  last two columns should be latitude and longitude
	kdtree: KDTree fitted with Craigslist data (Pandas DataFrame)
	data: Data used to fit the KDTree to be used to find comps
	model: predictive model for Craigslist data


	OUTPUT: predicted rent for the given point (float)
	'''
	gps = point[-2:]
	beds = point[0]
	#print "beds: ", beds
	ind, dist = kdtree.query_radius(gps, sort_results=True, 
				return_distance=True, r=0.0182) # queries a one-mile radius
	
	sorted_by_dist = data.iloc[ind[0]]
	comps = sorted_by_dist[sorted_by_dist['beds'] == beds]
	found_comps = comps.shape[0]
	#print "found_comps:",found_comps

	if found_comps == 0:
		print "No comps found for location: ", gps
		median = None
		mean = None
		prediction = None
	elif found_comps < num_comps:
		print "Only %s comps found for location: %s" % (found_comps, gps)
		median = np.median(comps['price'])
		mean = np.mean(comps['price'])
		no_gps = point[:-2]
		new_point = no_gps.append(pd.DataFrame([median, mean]))
		prediction = model.predict(new_point.T)[0]
	else:
		median = np.median(comps.iloc[:num_comps]['price'])
		mean = np.mean(comps.iloc[:num_comps]['price'])
		no_gps = point[:-2]
		new_point = no_gps.append(pd.DataFrame([median, mean]))
		prediction = model.predict(new_point.T)[0]

	return median, mean, prediction



def predict_break_even(a_prediction, c_prediction, c_opx = 0.37, a_opx = 0.32):
	'''
	INPUT:
	a_prediction: predicted nightly rate for a unit on airbnb
	c_prediction: predicted monthly rent for same unit
	c_opx: estimated operating cost for a long-term rental
	a_opx: estimated operating cost for a short-term rental

	OUTPUT:
	int (predicted number of days to break even)
	'''
	c_annual_revenue = c_prediction * 12 * (1 - c_opx)

	days = 0
	a_total_revenue = 0
	while a_total_revenue < c_annual_revenue:
		days += 1
		a_total_revenue += (a_prediction * (1 - a_opx))
	return days
	

if __name__=='__main__':
	data = pd.read_csv('../Data/Craigslist/cleaned_full2.csv')
	grid = pd.read_csv('../Data/all_resi_cl_feats.csv')
	grid = grid.drop('Unnamed: 0', axis=1)
	with open('../Models/Craigslist/kdtree_full2.pkl') as f:
		kdtree = pickle.load(f)

	with open('../Models/Craigslist/rf_craigslist.pkl') as f2:
		gb = pickle.load(f2)

	predictions = []
	medians = []
	means = []
	for i in xrange(len(grid)):
		if i % 200 == 0:
			print "Processing %s of %s" % (i, len(grid))
		point = grid.iloc[i]
		median, mean, prediction = find_comps_predict_one(point, kdtree, data, gb)
		predictions.append(prediction)
		medians.append(median)
		means.append(mean)

	#print "length of predictions: ", len(predictions)
	grid['comp_median_price'] = np.array(medians)
	grid['comp_mean_price'] = np.array(means)
	grid['predictions'] = np.array(predictions)

	grid.to_csv('../Data/Craigslist/all_resi_predictions.csv', encoding='utf-8')

