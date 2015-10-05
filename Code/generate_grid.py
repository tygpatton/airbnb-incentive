import numpy as np
import itertools


def generate_grid(min_lat, max_lat, min_long, max_long, 
	lat_range=200, long_range=200):
	'''
	INPUT:
	coordinates for a square
	OUTPUT:
	numpy array with a grid covering the square
	'''
	lats = np.linspace(min_lat, max_lat, num=lat_range)
	longs = np.linspace(min_long, max_long, num=long_range)
	lat_longs = itertools.product(lats, longs)
	return np.array(list(lat_longs))



if __name__=='__main__':
	'''
	The following coordinates are roughly the four corners
	of the City and Count of San Francisco
	SW: 37.693529, -122.507657
	SE: 37.696018, -122.374769
	NW: 37.809195, -122.525294
	NE: 37.813308, -122.357129
	'''
	min_lat = 37.693529
	max_lat = 37.813308
	min_long = -122.525294
	max_long = -122.357129
	grid = generate_grid(min_lat, max_lat, min_long, max_long)
	np.savetxt('../Data/sf_gps_grid.csv', grid, delimiter=',')

	with open('../Models/Craigslist/kdtree_full2.pkl') as f:
		kdtree = pickle.load(f)


