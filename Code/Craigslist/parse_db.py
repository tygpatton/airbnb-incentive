from pymongo import MongoClient
import pandas as pd
import time
import multiprocessing as mp
import numpy as np

def parse_info(df, cols):
  '''
  Input: Partially parsed pandas dataframe, column names for target dataframe
  Output: Fully parsed pandas dataframe with explicit features
  '''
  results_df            = pd.DataFrame(columns = cols, index=df['id'])
  results_df['id']      = df['id']
  results_df['heading'] = df['heading']
  results_df['body']    = df['body']
  results_df['price']   = df['price']
  results_df['flagged'] = df['flagged_status']
  results_df['date']    = pd.to_datetime(df['timestamp'], unit='s')
  
  #The conditionals here are to test if the field we're searching for is
  #present in a given listing. Some listings have missing fields.
  start = time.time()
  for i in xrange(len(df)):
    if i % 200 == 0 or i == 10:
      now = time.time()
      print "%s seconds elapsed" % (now - start)
      print "parsing %s of %s records" % (i, len(df))
      print '-' * 30

    loc_info = df.iloc[i]['location']
    if 'lat' in loc_info and 'long' in loc_info:
      results_df.ix[i,'lat'] = float(df.ix[i]['location']['lat'])
      results_df.ix[i,'long'] = float(df.ix[i]['location']['long'])
    else:
      results_df.ix[i,'lat'] = None
      results_df.ix[i,'long'] = None 

    if 'source_subloc' in df.iloc[i]['annotations']:
      results_df.ix[i, 'region'] = df.iloc[i]['annotations']['source_subloc']

    if 'source_neighborhood' in df.iloc[i]['annotations']:
      results_df.ix[i, 'neighborhood'] = df.iloc[i]['annotations']['source_neighborhood']
    else:
      results_df.ix[i, 'neighborhood'] = None

    if 'bedrooms' in df.iloc[i]['annotations']:
      #A few listings say 'Studio' rather than '0br' in the bedrooms field
      if df.iloc[i]['annotations']['bedrooms'][0] not in '0123456789':
         results_df.ix[i, 'beds'] = 0
      else:
        results_df.ix[i, 'beds'] = int(df.iloc[i]['annotations']['bedrooms'][0])
    else:
      results_df.ix[i, 'beds'] = None

    if 'bathrooms' in df.iloc[i]['annotations']:
      results_df.ix[i, 'baths'] = int(df.iloc[i]['annotations']['bathrooms'][0])
    else:
      results_df.ix[i, 'baths'] = None
    
    if 'street_parking' in df.iloc[i]['annotations']:
      results_df.ix[i, 'parking'] = 1
    elif 'carport' in df.iloc[i]['annotations']:
      results_df.ix[i, 'parking'] = 2
    elif 'off_street_parking' in df.iloc[i]['annotations']:
      results_df.ix[i, 'parking'] = 3
    elif 'attached_garage' in df.iloc[i]['annotations']:
      results_df.ix[i, 'parking'] = 4
    else:
      results_df.ix[i, 'parking'] = 0

    if 'sqft' in df.iloc[i]['annotations']:
      results_df.ix[i,'sqft'] = df.iloc[i]['annotations']['sqft']
    else:
      results_df.ix[i,'sqft'] = None

    if 'accuracy' in df.iloc[i]['location']:
      results_df.ix[i,'accuracy'] = df.iloc[i]['location']['accuracy']
    else:
      results_df.ix[i,'accuracy'] = None

    if 'apartment' in df.iloc[i]['annotations']:
      if df.iloc[i]['annotations']['apartment'] == 'YES':
        results_df.ix[i, 'apt'] = 1
      else:
        results_df.ix[i, 'apt'] = 0
    else:
      results_df.ix[i, 'apt'] = None

    if 'house' in df.iloc[i]['annotations']:
      if df.iloc[i]['annotations']['house'] == 'YES':
        results_df.ix[i, 'house'] = 1
      else:
        results_df.ix[i, 'house'] = 0
    else:
      results_df.ix[i, 'house'] = None

  return results_df


if __name__ == '__main__':
  db_client = MongoClient()

  db = db_client['craigslist']

  table = db['SF']

  query = {'annotations.source_subloc': 'sfc'} # limit to only City of SF

  cursor = table.find(query)

  df = pd.DataFrame(list(cursor))

  cols = ['id', 'heading', 'body', 'price', 'lat', 'long', 'region',  
            'neighborhood', 'beds', 'baths', 'parking', 'sqft', 'apt', 'accuracy',
            'date', 'flagged', 'house']

  # splits = np.array_split(df, 41)

  # for i in xrange(len(splits)):
  #   result = parse_info(splits[i], cols)
  #   result.to_csv('parsed/parsed2_%s' % i, encoding='utf-8')

  parsed = parse_info(df, cols)

  parsed.to_csv('parsed/parsed_final.csv', encoding='utf-8')






