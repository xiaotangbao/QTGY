# coding=utf-8
from data_scaling import *

# Training regressor with cross validation option and scaler option
def regressor_train(rgs, train_data, train_score, auxiliary_data=None, test_data=None, normalize=False, pca_n_components=None,
                    discrete_col=[],extreme_process='shrink', max_z_score=4, discrete_max_z_score=4, discrete_weight=1,
                    cv=None, **cv_params):

	# 创建scaler并传递参数
	scaler = YGTQ_Scaler(method=normalize, pca_n_components=pca_n_components, max_z_score=max_z_score, discrete_col=discrete_col,
	                     extreme_process=extreme_process, discrete_max_z_score=discrete_max_z_score, discrete_weight=discrete_weight)

	# 训练scaler
	train_data, train_score = scaler.fit_transform(train_data, train_score, auxiliary_data=auxiliary_data, test_data=test_data)

	print train_data.shape

	# 传递cv选项
	if cv:
		from sklearn.model_selection import GridSearchCV
		rgs = GridSearchCV(rgs, cv=cv, n_jobs=-1, **cv_params)

	# 训练模型
	rgs.fit(train_data, train_score)

	# 输出模型的训练MSE
	from sklearn.metrics import mean_squared_error
	print 'Train MSE: %f' % mean_squared_error(train_score, rgs.predict(train_data))

	return rgs, scaler


def regressor_predict(rgs, test_data, scaler):

	# 数据的标准化
	test_data = scaler.transform(test_data)
	# 使用模型预测并反标准化预测值
	return scaler.y_scaler.inverse_transform(rgs.predict(test_data))

def prediction_ensemble(score_list):
	import numpy as np
	return np.array(score_list).mean(0)

if __name__ == '__main__':
	import pandas as pd
	from sklearn.linear_model import Lasso
	train_data = pd.DataFrame([[1,2,3],[2,3,1]])
	rgs = Lasso()
	rgs, X_scaler = regressor_train(rgs, train_data, [1,1], normalize=True)
	print X_scaler