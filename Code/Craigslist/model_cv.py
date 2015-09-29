import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from clean_data import fill_bath_nulls

def fit_score_model(X_train, X_test, y_train, y_test, model):
	'''

	'''
	model.fit(X_train.values, y_train.values)
	y_pred = model.predict(X_test.values)

	mse = mean_squared_error(y_test.values, y_pred)
	mae = mean_absolute_error(y_test.values, y_pred)

	return mse, mae

if __name__ == '__main__':
	df = pd.read_csv('parsed/featurized_sample.csv')
	df = fill_bath_nulls(df)
	X = df[['beds', 'baths', 'parking', 'apt', 'house', 'comp_median_price']]
	y = df['price']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

	rf = RandomForestRegressor(n_estimators=100)
	lr = LinearRegression()
	gdbr = GradientBoostingRegressor()
	ada = AdaBoostRegressor()

	models = [rf, lr, gdbr, ada]

	for model in models:
		mse, mae = fit_score_model(X_train, X_test, y_train, y_test, model)
		print model.__class__.__name__
		print "MSE: ", mse
		print "MAE: ", mae
		print '-' * 30