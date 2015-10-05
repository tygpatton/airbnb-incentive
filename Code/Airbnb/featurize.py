import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import cPickle as pickle

def compute_comps_median(df, kdtree, num_comps=10, feature='bedrooms'):
	'''
	INPUT: parsed and cleaned DataFrame, fit KDTree
	OUTPUT: DataFrame with added column
	for median price of num_comps closest comps
	'''
	gps = df[['latitude', 'longitude']]

	# line 15 finds the 100 closest properties
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
		comp_trigger = df.iloc[i][feature]
		sorted_by_dist = df.iloc[ind[i][1:]] # excludes the queried point from the result
		sorted_comps = sorted_by_dist[sorted_by_dist[feature] == comp_trigger]
		found_comps = sorted_comps.shape[0]

		if found_comps == 0:
			print 'No comps found for record %s (%s: %s)' % (i, feature, comp_trigger)
			medians.append(np.nan)
			means.append(np.nan)
			comp_counts.append(0)
		elif found_comps < num_comps:
			print 'Only %s comps for record %s (%s: %s)' % \
						(found_comps, i, feature, comp_trigger)
			median = np.median(sorted_comps.iloc[0:,:]['price'])
			mean = np.mean(sorted_comps.iloc[0:,:]['price'])
			medians.append(median)
			means.append(mean)
			comp_counts.append(found_comps)
			if found_comps < 3:
				fewer_than_three += 1
		else:
			median = np.median(sorted_comps.iloc[0:num_comps]['price'])
			mean = np.mean(sorted_comps.iloc[0:num_comps]['price'])
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

def test_num_comps(row, threshold, median_dict, feat='bedrooms'):
	'''
	Small function to be used in add_city_median
	'''
	if row.comps_found < threshold:
		row.comp_median_price = median_dict[row[feat]]
		row.comp_mean_price = median_dict[row[feat]]	
	return row


def add_city_median(df, comps_threshold=5, feature='bedrooms'):
	'''
	INPUT: DataFrame with median price of comps computed
	OUTPUT: DataFrame with median price replaced with
	city-wide median per bedroom count for properties
	with fewer than comps_threshold comps
	'''
	unique = df[feature].unique()
	city_dict = {x: np.median(df[df[feature] == x].price) for\
					x in unique}
	df = df.apply(test_num_comps, threshold=comps_threshold,
				median_dict=city_dict, feat=feature, axis=1)

	return df

def get_clusters(df, k=30):

	gps = df[['latitude', 'longitude']]
	kmeans = KMeans(n_clusters=k)

	kmeans.fit(gps)

	cluster_centers = pd.DataFrame(kmeans.cluster_centers_, \
					columns=['Latitude', 'Longitude'])
	cluster_labels = kmeans.labels_

	df['cluster'] = cluster_labels

	return df


def compute_cluster_medians(df):

	prices = df['price']

	df['cluster_median_price'] = df['cluster'].map(lambda x: \
							np.median(prices[df['cluster'] == x]))

	cluster_centers_df = df[['cluster', 'cluster_median_price', 'latitude',\
							'longitude']].groupby('cluster').mean()

	cluster_centers_df.reset_index(inplace=True)

	cluster_centers_df['count'] = df.groupby('cluster').count()['latitude']

	df.drop(['latitude', 'longitude'], axis=1, inplace=True)

	return df, cluster_centers_df

def get_bed_dummies(df):
	'''
	Creates dummy variables for bed_type and appends them
	to the DataFrame
	'''
	bed_type_dummies = pd.get_dummies(df['bed_type'])
	bed_type_dummies.drop('Airbed', axis=1, inplace=True)
	df = pd.merge(df, bed_type_dummies, 
					right_index=True, left_index=True)
	df.drop('bed_type', axis=1, inplace=True)
	return df


if __name__ == '__main__':

	df = pd.read_csv('../../Data/Airbnb/cleaned_data2.csv')
	df = get_bed_dummies(df)
	with open('../../Models/Airbnb/kdtree.pkl') as f:
		kdtree = pickle.load(f)

	df2 = compute_comps_median(df, kdtree, num_comps=100, feature='bedrooms')
	df2 = add_city_median(df2, feature='bedrooms')
	df2.to_csv('../../Data/Airbnb/featurized.csv', encoding='utf-8')
