import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def clean_data(df, return_large_df=False):
	'''
	INPUT:
	df: pandas DataFrame of raw InsideAirbnb Listings Data
	OUTPUT:
	cleaned pandas dataframe
	'''
	# limit data to only entire home/apt listings
	df = df[df['room_type'] == 'Entire home/apt']

	# strip away the '$' from the price cols and convert to float
	df['price'] = df['price'].map(lambda x: float(x.translate(None,'$,')))
	df['extra_people'] = df['extra_people'].map(lambda x: \
								float(x.translate(None,'$,')))

	# uncomment if using monthly price
	# listings2_df['monthly_price'] = listings2_df['monthly_price']\
				# .map(lambda x: float(x.translate(None,'$,')) if x >= 0 else x)
	
	# select only numerical features and bed_type to use later
	df = df[['host_is_superhost', 'latitude', 'longitude', 
					'accommodates', 'bathrooms', 'bedrooms', 'beds', 
					'price','guests_included', 'extra_people', 
					'minimum_nights', 'maximum_nights',
					'availability_30', 'number_of_reviews', 'bed_type']]

	# select features user will be able to input
	select = df[['accommodates', 'bathrooms', 'bedrooms', 'beds', 
					'price', 'guests_included', 'extra_people', 
					'minimum_nights', 'maximum_nights', 'latitude',
					'longitude', 'bed_type']]

	# engineer feature 'accomodated per bed'
	select['acc_per_bed'] = select['accommodates'] / select['beds']

	# replace inf
	select['acc_per_bed'].replace(np.inf, select['accommodates'], inplace=True)

	select = select.drop(['accommodates', 'beds'], axis=1)

	if return_large_df:
		return df, select
	else:
		return select


def fill_nulls(df, feature):
	'''
	INPUT: 
	df: Pandas DataFrame with 'accommodates' field
	feature: column with nulls to be filled

	Replaces nulls with median value of 'feature' for listings
	with same # people accommodated. Alters DataFrame in place.

	OUTPUT:
	None
	'''
	# computes median value of 'feature' for each unique value of 'accommodates'
	median_dict = {acc: np.median(df[df['accommodates']==acc][feature])\
						for acc in df['accommodates'].unique()}

	nulls_df = df[df[feature].isnull()]

	nulls_df[feature] = nulls_df['accommodates'].map(lambda acc:\
						median_dict[acc])

	df[feature].fillna(value=nulls_df[feature], inplace=True)

	return None

def remove_outliers(df):
	'''
	INPUT:
	pandas DataFrame

	OUTPUT:
	pandas DataFrame with outliers removed
	Outliers defined as price < 50 and price > 1500
	'''
	return df[((df['price'] < 1500) & (df['price'] > 50))]


if __name__ == '__main__':
	df = pd.read_csv('../../Data/Airbnb/listings 2.csv')

	to_fill = ['bedrooms', 'bathrooms', 'beds']

	for feature in to_fill:
		fill_nulls(df, feature)

	X = clean_data(df)

	X = remove_outliers(X)

	X.to_csv('../../Data/Airbnb/cleaned_data2.csv')

