import cPickle as pickle
import pandas as pd
import numpy as np


def compute_comps_median(df, kdtree, num_comps=10, train=True):
	'''
	INPUT: parsed and cleaned DataFrame, fit KDTree
	OUTPUT: DataFrame with added column
	for median price of num_comps closest comps
	'''
	gps = df[['lat', 'long']]

	# finds the 100 closest properties
	#dist, ind = kdtree.query(gps, k=100)
	
	# finds all properties within 1 mile radius
	ind, dist = kdtree.query_radius(gps, sort_results=True, 
					return_distance=True, r=0.0182) # lat, long radius

	full_comps = 0
	fewer_than_three = 0
	medians = []
	means = []
	comp_counts = []

	for i in xrange(len(ind)):
		beds = df.iloc[i].beds
		if train:
			sorted_by_dist = df.iloc[ind[i][1:]] # excludes the queried point from the result
		else:
			sorted_by_dist = df.iloc[ind[i]] 
		sorted_comps = sorted_by_dist[sorted_by_dist.beds == beds]
		found_comps = sorted_comps.shape[0]

		if found_comps == 0:
			print 'No comps found for record %s (%s beds)' % (i, beds)
			medians.append(np.nan)
			means.append(np.nan)
			comp_counts.append(0)
		elif found_comps < num_comps:
			print 'Only %s comps for record %s (%s beds)' % \
						(found_comps, i, beds)
			median = np.median(sorted_comps.iloc[0:,:]['price'])
			mean = np.mean(sorted_comps.iloc[0:,:]['price'])
			medians.append(median)
			means.append(mean)
			comp_counts.append(found_comps)
			if found_comps < 3:
				fewer_than_three += 1
		else:
			median = np.median(sorted_comps.iloc[0:num_comps]['price'])
			mean = np.mean(sorted_comps.iloc[0:num_comps,:]['price'])
			medians.append(median)
			means.append(mean)
			comp_counts.append(num_comps)
			full_comps += 1
	print '-'*30
	print "Done. %s of %s had %s or more comps" % \
				(full_comps, df.shape[0], num_comps)
	print "%s of %s had fewer than three comps" % \
				(fewer_than_three, df.shape[0])
	df['comp_median_price'] = np.array(medians)
	df['comp_mean_price'] = np.array(means)
	df['comps_found'] = np.array(comp_counts)
	df['fewer_than_five'] = df.comps_found.map(lambda comps: 1 if comps < 5 else 0)

	return df

def test_num_comps(row, threshold, median_dict):
	'''
	Small function to be used in add_city_median
	'''
	if row.comps_found < threshold:
		row.comp_median_price = median_dict[row.beds]
		row.comp_mean_price = median_dict[row.beds]
	return row


def add_city_median(df, comps_threshold=5):
	'''
	INPUT: DataFrame with median price of comps computed
	OUTPUT: DataFrame with median price replaced with
	city-wide median per bedroom count for properties
	with fewer than comps_threshold comps
	'''
	unique_beds = df.beds.unique()
	city_dict = {bed: np.median(df[df.beds == bed].price) for\
					bed in unique_beds}
	df = df.apply(test_num_comps, threshold=comps_threshold,
				median_dict=city_dict, axis=1)

	return df

if __name__ == '__main__':
	# df = pd.read_csv('../../Data/Craigslist/cleaned_full2.csv')
	
	with open('../../Models/Craigslist/kdtree_full2.pkl') as f:
		kdtree = pickle.load(f)

	# comp_amts = [10, 20, 50, 100]

	# for n in comp_amts:
	# 	featurized = compute_comps_median(df, kdtree, num_comps=n)
	# 	featurized = add_city_median(featurized, comps_threshold=1)
	# 	featurized.to_csv('../../Data/Craigslist/comp_variants/beds_%s_wo_city.csv' % n,
	# 					encoding='utf-8')
	# 	featurized = add_city_median(featurized, comps_threshold=5)

	# 	featurized.to_csv('../../Data/Craigslist/comp_variants/beds_%s_w_city.csv' % n,
	# 					encoding='utf-8')

	df = pd.read_csv('../../Data/Craigslist/sf_grid_points.csv')
	featurized = compute_comps_median(df, kdtree, train=False)
	featurized.to_csv('../../Data/Craigslist/sf_grid_predicted.csv', encoding='utf-8')



