import pandas as pd
from sklearn.neighbors import KDTree
import cPickle as pickle

def fit_tree(df):
	'''
	Fits a KDTree to geographic data
	'''
	gps = df[['latitude', 'longitude']]
	kdtree = KDTree(gps, leaf_size=500)
	return kdtree


if __name__ == '__main__':
	df = pd.read_csv('../../Data/Airbnb/cleaned_data.csv')

	tree = fit_tree(df)

	with open('../../Models/Airbnb/kdtree.pkl', 'w') as f:
		pickle.dump(tree, f)
