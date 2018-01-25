import cPickle
import pandas as pd
import matplotlib.pyplot as plt
from regressor import *
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LassoCV, Lasso

train_data, train_score, test_data, test_score = cPickle.load(open('offline_data.pkl'))
print train_data.shape, test_data.shape

from categorical_processing import *
full_data = pd.concat([train_data, test_data], axis=0)

full_data, new_col_name = categorical_processing(full_data, method='complex')

train_data = full_data.loc[train_data.index,]
test_data = full_data.loc[test_data.index,]

print train_data.shape, test_data.shape


train_mse = []
test_mse = []
n_estimators_list = range(20,320,20)
warm_start = True
scaler = YGTQ_Scaler(method='categorical', max_z_score=2, discrete_col=new_col_name, discrete_max_z_score=5,
                     discrete_weight=1)
train_data, train_score = scaler.fit_transform(train_data, train_score, auxiliary_data=None, test_data=test_data)
test_data = scaler.transform(test_data)

regressor = BaggingRegressor(Lasso(normalize=False, alpha=0.002), n_estimators=20,warm_start=warm_start,n_jobs=-1,max_features=0.7)

for n_estimators in n_estimators_list:
	regressor.set_params(n_estimators=n_estimators)
	regressor.fit(train_data, train_score)
	train_pred = regressor.predict(train_data)
	pred_score = scaler.y_scaler.inverse_transform(regressor.predict(test_data))

	# Calculate mean squared error
	from sklearn.metrics import mean_squared_error
	print 'Train MSE: %f' % mean_squared_error(train_score, train_pred)
	print 'Test MSE: %f' % mean_squared_error(test_score, pred_score)
	train_mse.append(mean_squared_error(train_score, train_pred))
	test_mse.append(mean_squared_error(test_score, pred_score))


plt.plot(n_estimators_list,train_mse)
plt.plot(n_estimators_list,test_mse)
plt.legend(['train_mse','test_mse'])
plt.show()