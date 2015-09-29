import pandas as pd

def drop_dupes(df):
	'''
	INPUT: pandas DataFrame of parsed data
	OUTPUT: cleaned data
	'''
	# remove duplicates of exact location,
	# beds, and price
	df = df.drop_duplicates(['lat', 'long', 'beds', 'price'], 
							take_last=True)
	return df

def remove_outliers(df, min_price=500, max_price=10000):
	'''
	INPUT: pandas DataFrame
	OUTPUT: DataFrame with price outliers removed
	'''

	return df[(df.price > min_price) & (df.price < max_price)]

def remove_outliers_per_beds(df):
	# define reasonable rent range for each
	# unique number of bedrooms
	range_dict = {0: (0,6000), 1: (0, 6100), 2: (700, 8000), 
				3: (1000, 999999), 4: (1500, 999999), 
				5: (1500, 999999), 6: (2000, 999999),
				7: (2000, 999999), 8: (2000, 999999)}
	base = df[df.beds == 0]
	base = base[base.price < 6000]

	beds_list = df['beds'].unique()
	for beds in beds_list:
		filtered = df[df['beds'] == beds]
		filtered = filtered[(filtered.price > range_dict[beds][0]) & \
							(filtered.price < range_dict[beds][1])]
		base = pd.concat([base, filtered])
	return base

def drop_columns(df):
	'''
	INPUT: pandas DataFrame of parsed data
	OUTPUT: pandas DataFrame with old index removed
	'''
	df.set_index('id', inplace=True)
	df.drop(['Unnamed: 0'], axis=1, inplace=True)
	df.reset_index(inplace=True)
	return df

def drop_nulls(df):
	'''
	INPUT: pandas DataFrame
	OUTPUT: pandas DataFrame with nulls dropped
	for columns: beds, price
	'''
	return df.dropna(subset=['beds', 'price'], how='any')

def fill_bath_nulls(df):
	'''
	INPUT: pandas DataFrame
	OUTPUT: pandas DataFrame with null values filled
	'''
	bath_dict = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 4, 8: 5}

	nulls_df = df[df.baths.isnull()]
	nulls_df.baths = nulls_df.beds.map(lambda beds:\
					bath_dict[beds])

	df['baths'].fillna(value=nulls_df['baths'], inplace=True)

	return df

def main():
	df = pd.read_csv('parsed/parsed2_0')
	for i in xrange(1, 41):
		small = pd.read_csv('parsed/parsed2_%s' % i)
		df = pd.concat([df, small])


	df = drop_dupes(df)
	df = drop_nulls(df)
	df = remove_outliers(df)
	df = remove_outliers_per_beds(df)
	df = drop_columns(df)
	df = fill_bath_nulls(df)
	return df

if __name__ == '__main__':
	df = main()

	df.to_csv('parsed/cleaned_full2.csv', encoding='utf-8')
