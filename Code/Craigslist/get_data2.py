import requests
import json
from pymongo import MongoClient
from datetime import timedelta, date
import pandas as pd


'''
Script to get data from an Amazon S3 bucket
Heavily adapted from Jon Oleson's project 'Price My Rental':
https://github.com/jonoleson/PriceMyRental
'''

db_client = MongoClient()
db = db_client['craigslist']
table = db['SF']

def daterange(start_date, end_date):
	'''
	INPUT: A start date and end date in datetime format
	OUTPUT: A range of dates, in 'YYYY-MM-DD' format
	'''
	for n in range(int((end_date - start_date).days)):
		single_date = start_date + timedelta(n)
		yield single_date.strftime('%Y-%m-%d')


def get_data(start_date, end_date, one_city=False, print_urls=False):
	'''
	Input: Start and end dates, in datetime format
	Output: None

	'''
	# Get dataframe of cities
	cities = pd.read_csv('ongoing_cities.csv', header = False)

	# If True, parse only the data for San Francisco
	if one_city:
		cities = cities[(cities['city'] == 'San Francisco')]


	for i in xrange(len(cities)):
		if not one_city:
			city  = cities.city_code[i]
			state = cities.state_code[i]
		else:
			city  = cities.city_code.values[0]
			state = cities.state_code.values[0]

		# Insert json into MongoDB
		for date in daterange(start_date, end_date):
			done_parsing = False
			k=0

			while not done_parsing:
				url = 'https://s3-us-west-2.amazonaws.com/hoodsjson/%s/%s/%s/%s.html' \
		        		% (state, city, date, k)
			    # If print_urls==True, Print out the url of each json as 
			    # it's being parsed
				if print_urls:
					print url
				# get http request via request module
				r = requests.get(url)

				# check to see if valid url
				if r.status_code != 200:
					print 'No data on %s' % date
					break

			    # Test if response has any data (the last json file of a day, if 
			    # that day contains data at all, is always valid but is empty)
				if len(r.json()) > 0:
					table.insert(r.json())
					k += 1
				else:
					done_parsing = True

if __name__ == '__main__':
	start_date = date(2015, 6, 20)
	end_date   = date(2015, 9, 22)
	get_data(start_date, end_date, one_city=True, print_urls=True)








