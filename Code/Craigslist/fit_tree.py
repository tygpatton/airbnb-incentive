import pandas as pd
from sklearn.neighbors import KDTree
import cPickle as pickle

def fit_tree(df):
	'''
	Fits a KDTree to geographic data
	'''
	gps = df[['lat', 'long']]
	kdtree = KDTree(gps, leaf_size=500)
	return kdtree


if __name__ == '__main__':
	df = pd.read_csv('parsed/cleaned_full.csv')

	tree = fit_tree(df)

	with open('models/kdtree_full.pkl', 'w') as f:
		pickle.dump(tree, f)
